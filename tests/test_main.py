"""Check the application entry point and its project overview output."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def main_result():
    """Run the entry point once using the current Python interpreter."""
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(project_root / "main.py")],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_main_executes_without_errors(main_result):
    assert main_result.returncode == 0, main_result.stderr
    assert main_result.stderr == ""


def test_main_displays_project_title(main_result):
    assert "Adaptive Energy Brain" in main_result.stdout.splitlines()


def test_main_displays_component_status(main_result):
    lines = main_result.stdout.splitlines()
    assert "Implemented components:" in lines
    assert "Current phase:" in lines
    assert "Planned components:" in lines
    implemented_components = (
        "Data preparation",
        "Energy Simulation",
        "Rule-based Baselines",
        "GNN-Transformer Forecasting (trained and evaluated research pipeline)",
    )
    for component in implemented_components:
        assert f"- {component}" in lines

    assert (
        "- Deep Reinforcement Learning (next planned research phase; not implemented)"
        in lines
    )
    planned_components = (
        "Deep Reinforcement Learning",
        "Meta-Learning Policy Selector",
        "DRO/CVaR Risk Optimization",
        "Dashboard",
        "Optional RAG Assistant",
    )
    for component in planned_components:
        assert f"- {component}" in lines
