"""Acquire byte-preserved CSV files with size limits and provenance sidecars."""

from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

import yaml


DEFAULT_MAX_BYTES = 50 * 1024 * 1024
CHUNK_BYTES = 64 * 1024


class AcquisitionError(Exception):
    """An input, transfer, or publication failed without replacing a dataset."""


def _validate_url(url, resolve=False):
    if not isinstance(url, str) or any(c.isspace() for c in url):
        raise AcquisitionError("URL must be a nonempty HTTP/HTTPS URL.")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError as exc:
        raise AcquisitionError(f"Invalid URL: {exc}") from exc
    if (
        parts.scheme not in {"http", "https"}
        or not host
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or any(ord(c) < 32 for c in url)
    ):
        raise AcquisitionError("URL requires HTTP/HTTPS, a host, no credentials or fragment.")
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")):
        raise AcquisitionError("Only public HTTP/HTTPS hosts are allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise AcquisitionError("Only public HTTP/HTTPS addresses are allowed.")
    if resolve:
        try:
            addresses = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80))
        except OSError as exc:
            raise AcquisitionError(f"Cannot resolve public source host {host}: {exc}") from exc
        if not addresses or any(
            not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses
        ):
            raise AcquisitionError("Source host must resolve only to public addresses.")
    return parts


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_parts = _validate_url(newurl, resolve=True)
        if urlsplit(req.full_url).scheme == "https" and new_parts.scheme != "https":
            raise AcquisitionError("HTTPS-to-HTTP redirect refused.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_http(url, timeout):
    _validate_url(url, resolve=True)
    # Ignore environment proxies so public-host validation is not bypassed by one.
    return build_opener(ProxyHandler({}), _PublicRedirects()).open(url, timeout=timeout)


def validate_source(source):
    """Validate one source entry without network access or filesystem writes."""
    if not isinstance(source, dict):
        raise AcquisitionError("Each source must be a mapping.")
    allowed = {
        "id", "category", "type", "local_path", "download_url", "filename",
        "status", "sha256", "description", "expected_columns", "units",
        "time_resolution", "time_zone", "license",
    }
    if set(source) - allowed:
        raise AcquisitionError(f"Unknown source fields: {sorted(set(source) - allowed)}")
    if not isinstance(source.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", source["id"]):
        raise AcquisitionError("Source id must contain letters, digits, underscores or hyphens.")
    if source.get("category") not in {"solar", "wind", "demand", "weather", "grid_prices"}:
        raise AcquisitionError("Source category must be solar, wind, demand, weather or grid_prices.")
    if not isinstance(source.get("description"), str) or not source["description"].strip():
        raise AcquisitionError("Source description must be nonempty text.")
    columns = source.get("expected_columns")
    units = source.get("units")
    if (
        not isinstance(columns, list) or not columns
        or not all(isinstance(column, str) and column for column in columns)
        or len(set(columns)) != len(columns)
        or not isinstance(units, dict) or set(units) != set(columns)
        or not all(isinstance(unit, str) and unit for unit in units.values())
    ):
        raise AcquisitionError("expected_columns must be unique names with one unit per column.")
    name = source.get("filename")
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv", name, re.IGNORECASE)
        or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"}
        or re.fullmatch(r"(?:COM|LPT)[1-9]", name.split(".")[0], re.IGNORECASE)
    ):
        raise AcquisitionError("filename must be a safe CSV basename, without directories.")
    kind = source.get("type")
    if kind not in {"local_csv", "http"}:
        raise AcquisitionError("Source type must be 'local_csv' or 'http'.")
    if source.get("status") not in {"pending", "verified"}:
        raise AcquisitionError("Source status must be 'pending' or 'verified'.")
    for field in ("time_resolution", "time_zone", "license", "local_path", "download_url"):
        if field not in source or (source[field] is not None and not isinstance(source[field], str)):
            raise AcquisitionError(f"{field} must be text or null and must be present.")
    if source["status"] == "pending":
        if source["download_url"] is not None:
            raise AcquisitionError("Pending sources must have download_url: null.")
    elif not all(source[field] and source[field].strip() for field in ("time_resolution", "time_zone", "license")):
        raise AcquisitionError("Verified sources require resolution, time zone and license information.")
    elif kind == "local_csv":
        if not source["local_path"] or not source["local_path"].strip():
            raise AcquisitionError("Verified local sources require a local_path.")
        if Path(source["local_path"]).suffix.lower() != ".csv":
            raise AcquisitionError("Local input must have a .csv extension.")
        if source["download_url"] is not None:
            raise AcquisitionError("Local sources require download_url: null.")
    elif kind == "http":
        _validate_url(source["download_url"])
        if source["local_path"] is not None:
            raise AcquisitionError("HTTP sources require local_path: null.")
    checksum = source.get("sha256")
    if checksum is not None and (
        not isinstance(checksum, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum)
    ):
        raise AcquisitionError("sha256 must contain exactly 64 hexadecimal characters.")


def load_sources(config_path):
    """Read the documented YAML schema; return entries without acquiring them."""
    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AcquisitionError(f"Cannot read source configuration {path}: {exc}") from exc
    if not isinstance(config, dict) or set(config) != {"sources"}:
        raise AcquisitionError("Configuration must contain a single 'sources' list.")
    sources = config["sources"]
    if not isinstance(sources, list) or not sources:
        raise AcquisitionError("sources must be a nonempty list.")
    ids, names = set(), set()
    for source in sources:
        validate_source(source)
        if source["id"] in ids or source["filename"].lower() in names:
            raise AcquisitionError("Source ids and output filenames must be unique.")
        ids.add(source["id"])
        names.add(source["filename"].lower())
    return sources


def acquire_dataset(source, raw_dir, *, base_dir=None, dry_run=False,
                    allow_large=False, max_bytes=DEFAULT_MAX_BYTES, timeout=30):
    """Import/download a CSV and JSON sidecar, refusing existing destinations.

    Relative local paths are resolved against base_dir (the config directory in
    the CLI). Dry runs validate inputs but perform no DNS requests or writes.
    allow_large is explicit per-call approval to exceed the default size limit.
    """
    validate_source(source)
    if not isinstance(max_bytes, int) or max_bytes <= 0 or timeout <= 0:
        raise AcquisitionError("max_bytes and timeout must be positive.")
    if source["status"] == "pending":
        return {
            "source_id": source["id"], "status": "skipped", "dry_run": dry_run,
            "reason": "Source is pending: dataset, access and license have not been verified.",
        }
    raw_dir = Path(raw_dir).resolve()
    destination = raw_dir / source["filename"]
    metadata_path = raw_dir / (source["filename"] + ".metadata.json")
    if os.path.lexists(destination) or os.path.lexists(metadata_path):
        raise AcquisitionError(f"Destination or metadata already exists: {destination}")
    local_path = None
    known_size = None
    if source["type"] == "local_csv":
        local_path = Path(source["local_path"])
        if not local_path.is_absolute():
            local_path = Path(base_dir or Path.cwd()) / local_path
        local_path = local_path.resolve()
        if not local_path.is_file():
            raise AcquisitionError(f"Local CSV file does not exist: {local_path}")
        try:
            known_size = local_path.stat().st_size
        except OSError as exc:
            raise AcquisitionError(f"Cannot inspect local CSV: {exc}") from exc
        if not allow_large and known_size > max_bytes:
            raise AcquisitionError("File exceeds size limit; explicit --allow-large approval required.")
    plan = {
        "source_id": source["id"],
        "source_type": source["type"],
        "source_location": str(local_path) if local_path else source["download_url"],
        "destination": str(destination),
        "metadata_path": str(metadata_path),
        "dry_run": dry_run,
        "status": "planned" if dry_run else "acquired",
        "known_size_bytes": known_size,
        "max_bytes": None if allow_large else max_bytes,
    }
    if dry_run:
        return plan

    published = []
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".acquire-", dir=raw_dir) as temporary:
            staged = Path(temporary) / "payload"
            stream = local_path.open("rb") if local_path else _open_http(source["download_url"], timeout)
            with stream:
                final_url = None
                if local_path is None:
                    if stream.status != 200:
                        raise AcquisitionError(f"Expected HTTP 200, received {stream.status}.")
                    final_url = stream.geturl()
                    length = stream.headers.get("Content-Length")
                    try:
                        known_size = int(length) if length is not None else None
                    except ValueError as exc:
                        raise AcquisitionError("Invalid HTTP Content-Length.") from exc
                    if known_size is not None and known_size < 0:
                        raise AcquisitionError("Negative HTTP Content-Length.")
                if not allow_large and known_size is not None and known_size > max_bytes:
                    raise AcquisitionError("File exceeds size limit; explicit --allow-large approval required.")
                digest = hashlib.sha256()
                count = 0
                with staged.open("xb") as output:
                    while True:
                        # Read no more than one byte beyond the approved limit.
                        read_size = CHUNK_BYTES if allow_large else min(CHUNK_BYTES, max_bytes - count + 1)
                        chunk = stream.read(read_size)
                        if not chunk:
                            break
                        count += len(chunk)
                        if not allow_large and count > max_bytes:
                            raise AcquisitionError("Transfer exceeds size limit; explicit --allow-large approval required.")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            if count == 0:
                raise AcquisitionError("Source is empty; no dataset published.")
            if known_size is not None and count != known_size:
                raise AcquisitionError(f"Incomplete transfer: expected {known_size} bytes, received {count}.")
            checksum = digest.hexdigest()
            if source.get("sha256") and checksum != source["sha256"].lower():
                raise AcquisitionError("SHA-256 mismatch; no dataset published.")
            metadata = {
                **plan,
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                "file_size_bytes": count,
                "sha256": checksum,
                "expected_sha256": source.get("sha256"),
                "final_url": final_url,
                "source_config": source,
            }
            sidecar = Path(temporary) / "metadata"
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            # Same-filesystem hard links publish complete files atomically without
            # replacing another file, including when acquisition calls race.
            os.link(staged, destination)
            published.append(destination)
            os.link(sidecar, metadata_path)
            published.append(metadata_path)
        return metadata
    except (OSError, ValueError, HTTPException, HTTPError, URLError, AcquisitionError) as exc:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise AcquisitionError(f"Acquisition failed for {source['id']}: {exc}") from exc
