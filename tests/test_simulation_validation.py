"""Final verification must retain failures and never fabricate readiness."""

from types import SimpleNamespace

import pytest

from scripts import validate_simulation as audit


def test_required_edge_case_audit_checks_expected_physics():
    result = audit.edge_cases()
    assert len(result["cases"]) == 10
    assert all(c["status"] == "passed" for c in result["cases"])
    shortage = next(c for c in result["cases"] if c["name"] == "high demand with insufficient supply")
    assert shortage["actual"]["unserved"] == 400


def test_replay_failure_is_retained_even_when_test_suite_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "inventory", lambda path: {})
    monkeypatch.setattr(audit, "edge_cases", lambda: {"cases": []})
    monkeypatch.setattr(audit, "full_tests", lambda output: {"passed_count": 10, "module_counts": {}, "log": "fixture"})

    def missing(*args):
        raise ValueError("No approved processed data")

    monkeypatch.setattr(audit, "replay", missing)
    result = audit.validate(manifest="absent", assets="absent", rating_origin="hypothetical_example",
                            config="absent", costs="absent", hours=1, output=tmp_path / "audit")
    assert not result["passed"]
    assert result["assessment"].startswith("NOT VERIFIED")
    report = audit.render_report(result)
    assert "No approved processed data" in report
    assert "NOT VERIFIED" in report
    assert "## End-to-end episode" not in report


def test_failed_pytest_run_cannot_be_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="1 failed", stderr=""))
    with pytest.raises(ValueError, match="Full test suite failed"):
        audit.full_tests(tmp_path)
    assert (tmp_path / "pytest.log").read_text() == "1 failed"
