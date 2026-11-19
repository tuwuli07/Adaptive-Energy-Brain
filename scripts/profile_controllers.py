"""Phase 8.3 Controller Profiling CLI Script."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.profiling import run_profiling_suite


if __name__ == "__main__":
    run_profiling_suite()
