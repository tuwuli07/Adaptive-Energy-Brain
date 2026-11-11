"""Offline acquisition tests using local fixtures and simulated HTTP streams."""

import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import socket
from urllib.error import URLError

import pytest

from scripts.download_data import main
from src.data import acquisition as acq


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Tests must not access the network")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "original.csv"
    path.write_bytes(b"timestamp,power\r\n2026-01-01,1.5\r\n")
    return {
        "id": "sample", "category": "solar", "description": "Test-only fixture",
        "type": "local_csv", "local_path": str(path), "download_url": None,
        "filename": "sample.csv", "status": "verified",
        "expected_columns": ["timestamp", "power"],
        "units": {"timestamp": "UTC datetime", "power": "kW"},
        "time_resolution": "hourly", "time_zone": "UTC", "license": "Test fixture",
    }


def http_source(source):
    return {**source, "type": "http", "local_path": None,
            "download_url": "https://example.org/test.csv"}


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return "https://example.org/test.csv"


def test_project_configuration_has_five_pending_categories(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sources = acq.load_sources(root / "configs/data_sources.yaml")
    assert {s["category"] for s in sources} == {
        "solar", "wind", "demand", "weather", "grid_prices",
    }
    assert len(sources) == 5
    for source in sources:
        assert source["status"] == "pending"
        assert source["download_url"] is None
        for dry_run in (False, True):
            result = acq.acquire_dataset(source, tmp_path / "raw", dry_run=dry_run)
            assert result["status"] == "skipped"
            assert "pending" in result["reason"]
    assert not (tmp_path / "raw").exists()


def test_local_import_preserves_original_and_records_provenance(source, tmp_path):
    original = Path(source["local_path"]).read_bytes()
    raw = tmp_path / "raw"
    result = acq.acquire_dataset(source, raw)
    assert (raw / "sample.csv").read_bytes() == original
    assert Path(source["local_path"]).read_bytes() == original
    metadata = json.loads((raw / "sample.csv.metadata.json").read_text())
    assert metadata == result
    assert metadata["sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["file_size_bytes"] == len(original)
    assert metadata["source_config"] == source
    assert metadata["acquired_at_utc"].endswith("+00:00")


def test_local_dry_run_writes_nothing(source, tmp_path):
    raw = tmp_path / "raw"
    result = acq.acquire_dataset(source, raw, dry_run=True)
    assert result["status"] == "planned"
    assert not raw.exists()


def test_http_dry_run_never_opens_connection(source, tmp_path, monkeypatch):
    def fail(*args):
        raise AssertionError("dry run opened HTTP")
    monkeypatch.setattr(acq, "_open_http", fail)
    assert acq.acquire_dataset(http_source(source), tmp_path / "raw", dry_run=True)["dry_run"]
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("existing", ["sample.csv", "sample.csv.metadata.json"])
def test_existing_files_are_never_overwritten(source, tmp_path, existing):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / existing).write_bytes(b"keep me")
    with pytest.raises(acq.AcquisitionError, match="already exists"):
        acq.acquire_dataset(source, raw)
    assert (raw / existing).read_bytes() == b"keep me"
    assert len(list(raw.iterdir())) == 1


def test_missing_local_path_is_useful_error(source, tmp_path):
    source["local_path"] = str(tmp_path / "missing.csv")
    with pytest.raises(acq.AcquisitionError, match="does not exist"):
        acq.acquire_dataset(source, tmp_path / "raw", dry_run=True)


@pytest.mark.parametrize("url", [
    "not a URL", "ftp://example.org/a.csv", "https:///a.csv",
    "http://localhost/a.csv", "http://127.0.0.1/a.csv",
    "https://user:password@example.org/a.csv", "https://example.org:bad/a.csv",
])
def test_invalid_urls_rejected(source, tmp_path, url):
    candidate = {**http_source(source), "download_url": url}
    with pytest.raises(acq.AcquisitionError):
        acq.acquire_dataset(candidate, tmp_path / "raw", dry_run=True)


def test_mock_http_download(source, tmp_path, monkeypatch):
    body = b"a,b\n1,2\n"
    monkeypatch.setattr(acq, "_open_http", lambda *args: Response(body, len(body)))
    result = acq.acquire_dataset(http_source(source), tmp_path / "raw")
    assert result["file_size_bytes"] == len(body)
    assert (tmp_path / "raw/sample.csv").read_bytes() == body


@pytest.mark.parametrize("failure", ["network", "incomplete", "read_error", "oversized", "unknown_size", "checksum", "empty"])
def test_failed_transfers_never_publish(source, tmp_path, monkeypatch, failure):
    candidate = http_source(source)
    if failure == "network":
        def opening(*args):
            raise URLError("test connection failed")
    elif failure == "read_error":
        class Broken(Response):
            def read(self, size=-1):
                raise IncompleteRead(b"partial", 10)
        opening = lambda *args: Broken(b"data")
    else:
        body = b"" if failure == "empty" else b"a,b\n1,2\n"
        length = 99 if failure == "incomplete" else len(body)
        if failure == "unknown_size":
            length = None
        if failure == "checksum":
            candidate["sha256"] = "0" * 64
        opening = lambda *args: Response(body, length)
    monkeypatch.setattr(acq, "_open_http", opening)
    raw = tmp_path / "raw"
    kwargs = {"max_bytes": 2} if failure in {"oversized", "unknown_size"} else {}
    with pytest.raises(acq.AcquisitionError):
        acq.acquire_dataset(candidate, raw, **kwargs)
    assert not list(raw.iterdir())


def test_large_local_import_requires_explicit_approval(source, tmp_path):
    with pytest.raises(acq.AcquisitionError, match="allow-large"):
        acq.acquire_dataset(source, tmp_path / "raw", max_bytes=1)
    result = acq.acquire_dataset(source, tmp_path / "raw", max_bytes=1, allow_large=True)
    assert result["status"] == "acquired"


def test_metadata_publication_failure_preserves_concurrent_file(source, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    link = acq.os.link

    def conflicting_link(staged, target):
        if str(target).endswith(".metadata.json"):
            Path(target).write_bytes(b"concurrent metadata")
        link(staged, target)

    monkeypatch.setattr(acq.os, "link", conflicting_link)
    with pytest.raises(acq.AcquisitionError):
        acq.acquire_dataset(source, raw)
    assert not (raw / "sample.csv").exists()
    assert (raw / "sample.csv.metadata.json").read_bytes() == b"concurrent metadata"


def test_redirect_to_private_host_is_rejected(monkeypatch):
    from urllib.request import Request

    request = Request("https://example.org/test.csv")
    with pytest.raises(acq.AcquisitionError, match="public"):
        acq._PublicRedirects().redirect_request(
            request, None, 302, "Found", {}, "http://127.0.0.1/test.csv",
        )


def test_public_host_validation_rejects_private_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443)),
    ])
    with pytest.raises(acq.AcquisitionError, match="public addresses"):
        acq._validate_url("https://example.org/test.csv", resolve=True)


def test_invalid_config_and_missing_config(tmp_path):
    with pytest.raises(acq.AcquisitionError, match="Cannot read"):
        acq.load_sources(tmp_path / "missing.yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("sources: not-a-list\n")
    with pytest.raises(acq.AcquisitionError, match="nonempty list"):
        acq.load_sources(path)


def test_cli_skips_pending_config(capsys):
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count('"status": "skipped"') == 5
    assert "pending" in output


def test_cli_reports_missing_config(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "Cannot read source configuration" in capsys.readouterr().err
