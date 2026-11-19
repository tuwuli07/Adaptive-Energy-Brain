"""Phase 8.2 Research Dashboard Application Entrypoint.

Usage:
    python -m src.dashboard.app [--generate] [--serve] [--port PORT] [--open]
    python scripts/launch_dashboard.py
"""

import argparse
import http.server
from pathlib import Path
import socketserver
import sys
import webbrowser

from src.dashboard.generator import DEFAULT_BENCHMARK_DIR, generate_all


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Adaptive Energy Brain Research Dashboard")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate static figures and dashboard HTML from validated benchmark data",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve dashboard locally over HTTP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for local HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard HTML in default system browser",
    )
    args = parser.parse_args()

    dashboard_file = DEFAULT_BENCHMARK_DIR / "dashboard.html"

    # Always generate if dashboard does not exist yet or if --generate requested
    if args.generate or not dashboard_file.exists():
        print("Generating research dashboard and publication figures...")
        res = generate_all()
        print(f"Generated dashboard HTML: {res['dashboard_html']}")
        print(f"Generated {len(res['figures'])} figures.")

    if args.open:
        print(f"Opening {dashboard_file} in default browser...")
        webbrowser.open(dashboard_file.as_uri())

    if args.serve:
        serve_dir = DEFAULT_BENCHMARK_DIR
        port = args.port

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args_inner, **kwargs):
                super().__init__(*args_inner, directory=str(serve_dir), **kwargs)

        print(f"\nServing research dashboard at: http://localhost:{port}/dashboard.html")
        print("Press Ctrl+C to stop the server.")
        with socketserver.TCPServer(("", port), QuietHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")


if __name__ == "__main__":
    main()
