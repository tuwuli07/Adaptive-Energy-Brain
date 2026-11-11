"""Run registered baseline controllers on identical simulator episodes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_runner import main


if __name__ == "__main__":
    sys.exit(main())
