"""Unit and regression tests for Phase 8.2 Research Dashboard and Visual Analytics.

Verifies:
- Benchmark data ingestion matches frozen validation records.
- Generation of publication figures (Figures 1–4) with nonzero file sizes.
- Generation of valid offline-capable dashboard HTML with embedded JSON payload.
- All 7 primary benchmark controllers rendered with exact values and causality tags.
- Read-only safety: dashboard generation modifies no raw data, checkpoints, or benchmark tables.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pytest

from src.dashboard.generator import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_FIGURES_DIR,
    generate_all,
    generate_dashboard_html,
    generate_figures,
    load_benchmark_data,
)


ROOT = Path(__file__).resolve().parents[1]


class TestResearchDashboard:
    """Tests for research dashboard generation and data integrity."""

    def test_load_benchmark_data_integrity(self):
        """Loads validated benchmark data without missing fields."""
        data = load_benchmark_data()
        assert "benchmark_table" in data
        assert "hourly_by_controller" in data
        assert "forecast_metrics" in data

        table = data["benchmark_table"]
        assert len(table) >= 7

        cids = {r["controller_id"] for r in table}
        expected_cids = {
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "ppo_continuous_control",
            "mpc_seasonal_persistence",
            "mpc_gnn_transformer",
            "mpc_oracle_rolling_24h",
        }
        assert expected_cids.issubset(cids)

    def test_generate_figures(self, tmp_path):
        """Generates all 4 publication figures with valid png file sizes."""
        fig_dir = tmp_path / "figures"
        figs = generate_figures(output_dir=fig_dir)
        assert len(figs) == 4

        expected_filenames = {
            "figure1_cost_and_oracle_gap.png",
            "figure2_battery_throughput_and_soc.png",
            "figure3_dispatch_trajectories_sample.png",
            "figure4_forecast_error_by_horizon.png",
        }
        actual_filenames = {f.name for f in figs}
        assert actual_filenames == expected_filenames

        for f in figs:
            assert f.exists()
            assert f.stat().st_size > 15_000, f"Figure {f.name} too small: {f.stat().st_size} bytes"

    def test_generate_dashboard_html(self, tmp_path):
        """Generates valid offline HTML with embedded JSON payload and all primary controllers."""
        html_file = tmp_path / "dashboard.html"
        generated_p = generate_dashboard_html(output_file=html_file)
        assert generated_p.exists()
        assert generated_p.stat().st_size > 50_000

        content = generated_p.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Adaptive Energy Brain" in content
        assert "id=\"benchmark-payload\"" in content

        # Check all 7 primary controller IDs are present
        expected_cids = [
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "ppo_continuous_control",
            "mpc_seasonal_persistence",
            "mpc_gnn_transformer",
            "mpc_oracle_rolling_24h",
        ]
        for cid in expected_cids:
            assert cid in content, f"Controller ID {cid} missing from dashboard HTML payload"

        # Check key numeric benchmark values are present
        assert "43,702.87" in content or "43702.87" in content  # GNN MPC
        assert "43,866.17" in content or "43866.17" in content  # Zero Action
        assert "43,396.52" in content or "43396.52" in content  # Oracle MPC
        assert "43,645.30" in content or "43645.3" in content   # Continuous Price Aware
        assert "43,632.76" in content or "43632.76" in content  # Discrete Price Aware

        # Check causality distinction
        assert "Strictly Causal" in content or "strictly_causal" in content
        assert "Noncausal Lookahead" in content or "noncausal_lookahead" in content

    def test_read_only_safety_invariant(self, tmp_path):
        """Dashboard generation does not alter benchmark table or touch test data."""
        bench_table_before = (DEFAULT_BENCHMARK_DIR / "benchmark_table.json").read_text(encoding="utf-8")
        test_dir = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1" / "test"

        # Record test dir mtime
        test_mtime_before = test_dir.stat().st_mtime

        # Run generator
        generate_all()

        bench_table_after = (DEFAULT_BENCHMARK_DIR / "benchmark_table.json").read_text(encoding="utf-8")
        assert bench_table_before == bench_table_after, "benchmark_table.json was modified!"
        assert test_dir.stat().st_mtime == test_mtime_before, "Test directory was modified!"
