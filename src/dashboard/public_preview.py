"""Bounded public-demo gateway: built UI + selected local API routes only.

Run with python -m src.dashboard.public_preview. Tunnel port 8787, never the
repository, development server, backend port, or Ollama port directly.
"""
import http.client
import hmac
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
READ_ROUTES = {
    "/api/health", "/api/overview", "/api/energy/timeseries",
    "/api/forecast/metrics", "/api/battery/dispatch", "/api/controllers",
    "/api/controllers/comparison", "/api/research/results", "/api/risk/results",
    "/api/adaptive/results", "/api/adaptive/trajectory", "/api/assistant/status",
    "/api/simulations/export", "/api/datasets",
}
PAGE_ROUTES = {"/", "/data", "/energy", "/forecasting", "/battery", "/adaptive", "/comparison", "/risk", "/explorer", "/research", "/about", "/assistant"}
DATASET_ROUTE = re.compile(r"/api/datasets/[a-f0-9]{32}(?:/(?:analysis|mapping))?\Z")
MODEL_SLOT = threading.BoundedSemaphore(1)
RATE_LOCK = threading.Lock()
RATE = defaultdict(deque)


def gemini_preview_enabled(environ=None):
    env = os.environ if environ is None else environ
    return (
        env.get("ENERGY_ASSISTANT_PROVIDER", "").strip().lower() == "gemini"
        and bool(env.get("GEMINI_API_KEY", "").strip())
        and len(env.get("ENERGY_ASSISTANT_ACCESS_KEY", "")) >= 32
    )


def valid_gemini_access_key(authorization, environ=None):
    env = os.environ if environ is None else environ
    expected = env.get("ENERGY_ASSISTANT_ACCESS_KEY", "")
    scheme, separator, supplied = authorization.partition(" ")
    return (
        gemini_preview_enabled(env)
        and scheme.lower() == "bearer"
        and bool(separator)
        and hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    )


def allowed_api(method, path):
    if method == "GET":
        return path in READ_ROUTES or bool(DATASET_ROUTE.fullmatch(path) and not path.endswith("/mapping"))
    return method == "POST" and (path in ("/api/datasets/upload", "/api/assistant/chat") or bool(DATASET_ROUTE.fullmatch(path) and path.endswith("/mapping")))


def safe_static(path, root=DIST):
    path = unquote(path)
    if path.rstrip("/") in PAGE_ROUTES or path == "/":
        return root / "index.html"
    if not path.startswith("/assets/") or "\\" in path or ".." in path.split("/"):
        return None
    target = (root / path.lstrip("/")).resolve()
    return target if target.is_relative_to(root.resolve()) and target.is_file() else None


class PreviewHandler(BaseHTTPRequestHandler):
    server_version = "EnergyPreview"

    def reply(self, status, content, content_type="application/json; charset=utf-8"):
        if isinstance(content, dict):
            content = json.dumps(content).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.proxy()
        else:
            file = safe_static(path)
            if file is None or not file.is_file():
                self.reply(404, {"error": "Not found"})
            else:
                self.reply(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")

    def do_POST(self):
        self.proxy()

    def proxy(self):
        parsed = urlsplit(self.path)
        if not allowed_api(self.command, parsed.path):
            return self.reply(404, {"error": "Endpoint unavailable in public preview"})
        origin = self.headers.get("Origin")
        if origin:
            host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host")
            if urlsplit(origin).netloc != host or urlsplit(origin).scheme not in ("http", "https"):
                return self.reply(403, {"error": "Cross-site requests are not allowed"})
        for key, values in parse_qs(parsed.query).items():
            if key == "controller_id" and any(not re.fullmatch(r"[a-z][a-z0-9_]{0,80}", value) for value in values):
                return self.reply(400, {"error": "Invalid strategy"})
            if key == "partition" and any(value not in ("test", "validation", "val") for value in values):
                return self.reply(400, {"error": "Invalid study partition"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "Invalid request length"})
        limit = 10 * 1024 * 1024 + 65536 if parsed.path.endswith("/upload") else 100000
        if length < 0 or length > limit or self.headers.get("Transfer-Encoding"):
            return self.reply(413, {"error": "Request exceeds the size limit"})
        self.connection.settimeout(30)
        body = self.rfile.read(length) if length else None
        slot = False
        if parsed.path == "/api/assistant/chat":
            try:
                data = json.loads(body or b"{}")
                if not isinstance(data, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                return self.reply(400, {"error": "Invalid chat request"})
            if data.get("provider") == "gemini":
                if not gemini_preview_enabled():
                    return self.reply(503, {"error": "Gemini is not enabled for this hosted assistant."})
                if not valid_gemini_access_key(self.headers.get("Authorization", "")):
                    return self.reply(401, {"error": "Enter a valid assistant access key to use Gemini."})
            else:
                # The public preview does not accept arbitrary providers or model
                # downloads. The local backend enforces installed model names.
                data["provider"] = "ollama"
            body = json.dumps(data).encode()
            client = self.headers.get("CF-Connecting-IP", self.client_address[0])
            now = time.monotonic()
            with RATE_LOCK:
                if len(RATE) > 10000:
                    RATE.clear()
                hits = RATE[client]
                while hits and now - hits[0] > 60:
                    hits.popleft()
                if len(hits) >= 6:
                    return self.reply(429, {"error": "Please wait a minute before asking more questions."})
                hits.append(now)
            slot = MODEL_SLOT.acquire(blocking=False)
            if not slot:
                return self.reply(429, {"error": "The local assistant is answering another visitor. Please try again shortly."})
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=110)
        try:
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json"), "Origin": "http://127.0.0.1:8000"}
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            if parsed.path == "/api/health" and response.status == 200:
                payload = json.dumps({"status": "online", "service": "Adaptive Energy Brain", "mode": "public_preview"}).encode()
            if parsed.path == "/api/assistant/status" and response.status == 200:
                data = json.loads(payload)
                gemini_enabled = gemini_preview_enabled()
                data.update({
                    "default_provider": "gemini" if gemini_enabled else "ollama",
                    "gemini_configured": gemini_enabled,
                    "assistant_access_required": gemini_enabled,
                    "public_preview": True,
                })
                payload = json.dumps(data).encode()
            if response.status >= 500:
                payload = json.dumps({"error": "The local service is unavailable. Please try again shortly."}).encode()
            self.reply(response.status, payload, response.getheader("Content-Type", "application/json"))
        except (OSError, http.client.HTTPException):
            self.reply(502, {"error": "The local backend is unavailable."})
        finally:
            conn.close()
            if slot:
                MODEL_SLOT.release()


def main():
    if not (DIST / "index.html").is_file():
        raise SystemExit("Build the frontend before starting the public preview.")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer((host, port), PreviewHandler)
    print(f"Built application preview: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
