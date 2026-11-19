"""Start the API and public preview gateway for a hosted Render web service."""

from pathlib import Path
import os
import subprocess
import sys
import time

import urllib.request


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

api_process = subprocess.Popen(
    [sys.executable, "-m", "src.dashboard.api", "8000"],
    cwd=ROOT,
)


def wait_for_api(timeout_seconds: int = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if api_process.poll() is not None:
            raise SystemExit(f"Backend exited with code {api_process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise SystemExit("Backend did not become ready within 30 seconds")


try:
    wait_for_api()
    os.environ.setdefault("HOST", "0.0.0.0")
    preview_process = subprocess.Popen(
        [sys.executable, "-m", "src.dashboard.public_preview"],
        cwd=ROOT,
    )
    return_code = preview_process.wait()
    raise SystemExit(return_code)
except BaseException:
    api_process.terminate()
    api_process.wait(timeout=10)
    raise
