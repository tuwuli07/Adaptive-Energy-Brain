"""Convenience script to launch the Phase 8.2 research dashboard."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.app import main


if __name__ == "__main__":
    main()
