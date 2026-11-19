"""CSV intake, numerical accounting and real HTTP integration regression tests."""
import hashlib
from http.client import HTTPConnection
from http.server import HTTPServer
import json
from pathlib import Path
import threading

import pytest

from src.data.user_datasets import DatasetStore, DatasetError, parse_csv, analyze, MAX_BYTES
from src.dashboard import api

VALID = b"timestamp,load,solar,wind,charge\n2026-01-01T00:00:00Z,10,2,1,50\n2026-01-01T00:30:00Z,20,4,2,60\n2026-01-01T02:00:00Z,30,6,3,70\n"
MAPPING = {"timestamp": {"column": "timestamp", "unit": "ISO8601"}, "demand_kw": {"column": "load", "unit": "kW"}, "solar_kw": {"column": "solar", "unit": "kW"}, "wind_kw": {"column": "wind", "unit": "kW"}, "battery_soc": {"column": "charge", "unit": "%"}}


def result(raw=VALID, mapping=MAPPING, **kwargs):
    return analyze(parse_csv(raw, "sample.csv")[1], mapping, confirm_conversions=True, **kwargs)


def test_irregular_integrals_and_denominators():
    data = result()
    summary = data["summary"]
    assert summary["energy_kwh"]["demand_kw"] == 35  # 10*0.5 + 20*1.5
    assert summary["average_kw"]["demand_kw"] == 17.5
    assert summary["peak_kw"]["demand_kw"] == 30
    assert summary["renewable_generation_kwh"] == 10.5
    assert summary["renewable_to_demand_pct"] == 30
    assert summary["irregular"] is True
    assert summary["duration_hours"] == 2
    assert data["records"][0]["battery_soc"] == .5


@pytest.mark.parametrize("raw,name", [(b"", "x.csv"), (b" \n", "x.csv"), (b"a,b\n", "x.csv"), (b"a,a\n1,2", "x.csv"), (b"a,b\n1,2,3", "x.csv"), (b'a,b\n"1,2', "x.csv"), (b"a,b\n1,2", "x.pkl"), (b"a,b\n1,\x00", "x.csv"), (b"a,b\n1,\xff", "x.csv")])
def test_malformed_csv(raw, name):
    with pytest.raises(DatasetError):
        parse_csv(raw, name)


def test_byte_limit_and_row_limit(monkeypatch):
    with pytest.raises(DatasetError) as caught:
        parse_csv(b"a" * (MAX_BYTES + 1), "x.csv")
    assert caught.value.status == 413
    monkeypatch.setattr("src.data.user_datasets.MAX_ROWS", 2)
    with pytest.raises(DatasetError, match="100,000"):
        parse_csv(VALID, "x.csv")


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-1", "1e1000"])
def test_invalid_numbers(value):
    with pytest.raises(DatasetError) as caught:
        result(VALID.replace(b",10,", f",{value},".encode()))
    assert caught.value.details["errors"][0]["row"] == 2


@pytest.mark.parametrize("stamp", [b"", b"2026-13-01T00:00:00Z", b"yesterday", b"2026-01-01T00:00:00"])
def test_invalid_timestamps(stamp):
    with pytest.raises(DatasetError):
        result(VALID.replace(b"2026-01-01T00:00:00Z", stamp))


def test_explicit_naive_utc_and_offset_normalization():
    raw = VALID.replace(b"2026-01-01T00:00:00Z", b"2026-01-01T00:00:00")
    assert result(raw, assume_utc=True)["summary"]["duration_hours"] == 2
    raw = VALID.replace(b"2026-01-01T00:00:00Z", b"2026-01-01T06:00:00+06:00")
    assert result(raw)["summary"]["duration_hours"] == 2


def test_duplicate_timestamps_are_blocked_including_equivalent_offsets():
    raw = VALID.replace(b"2026-01-01T00:30:00Z", b"2026-01-01T06:00:00+06:00")
    with pytest.raises(DatasetError) as caught:
        result(raw)
    assert caught.value.details["duplicate_timestamps"] == 1


def test_missing_optional_columns_and_missing_intervals():
    mapping = {k: v for k, v in MAPPING.items() if k in ("timestamp", "demand_kw")}
    data = result(VALID.replace(b",10,", b",,"), mapping)
    assert data["summary"]["missing_values"]["demand_kw"] == 1
    assert data["summary"]["energy_kwh"]["demand_kw"] == 30
    assert data["summary"]["covered_hours"]["demand_kw"] == 1.5
    assert data["summary"]["renewable_generation_kwh"] is None
    assert "solar_kw" not in data["records"][0]


def test_unit_mapping_and_conversion_requires_confirmation():
    _, rows = parse_csv(VALID, "x.csv")
    with pytest.raises(DatasetError, match="Confirm"):
        analyze(rows, MAPPING)
    mapping = {"timestamp": MAPPING["timestamp"], "demand_kw": {"column": "load", "unit": "W"}}
    data = analyze(rows, mapping, confirm_conversions=True)
    assert data["summary"]["energy_kwh"]["demand_kw"] == pytest.approx(.035)
    mapping["demand_kw"]["unit"] = "MW"
    assert analyze(rows, mapping, confirm_conversions=True)["summary"]["energy_kwh"]["demand_kw"] == 35000
    mapping["demand_kw"]["unit"] = "kWh"
    with pytest.raises(DatasetError, match="supported unit"):
        analyze(rows, mapping)


@pytest.mark.parametrize("mapping", [None, {}, {"unknown": {}}, {"demand_kw": MAPPING["demand_kw"]}, {"timestamp": MAPPING["timestamp"]}])
def test_required_mapping(mapping):
    with pytest.raises(DatasetError):
        result(mapping=mapping)


def test_safe_storage_selection_and_persistence(tmp_path):
    store = DatasetStore(tmp_path)
    meta = store.upload(VALID, "../../outside.csv")
    assert meta["name"] == "outside.csv"
    assert not (tmp_path.parent / "outside.csv").exists()
    assert (tmp_path / meta["id"] / "original.csv").read_bytes() == VALID
    assert meta["sha256"] == hashlib.sha256(VALID).hexdigest()
    with pytest.raises(DatasetError):
        store.analysis(meta["id"])
    store.apply_mapping(meta["id"], {"mapping": MAPPING, "confirm_conversions": True})
    restored = DatasetStore(tmp_path)
    assert restored.list()[0]["status"] == "ready"
    assert restored.get(meta["id"])["mapping"] == MAPPING
    assert restored.analysis(meta["id"])["summary"]["energy_kwh"]["demand_kw"] == 35
    for invalid in ("../", "../../configs", "x", "f" * 32):
        with pytest.raises(DatasetError):
            store.get(invalid)


def test_temporary_storage_is_disclosed_and_session_scoped(monkeypatch):
    monkeypatch.delenv("ENERGY_DATASET_STORAGE", raising=False)
    first, second = DatasetStore(), DatasetStore()
    assert first.info()["mode"] == "temporary"
    assert "restart" in first.info()["message"]
    first.upload(VALID, "x.csv")
    assert second.list() == []


def test_compatibility_never_promises_model_inference():
    compatibility = result()["compatibility"]
    assert compatibility["forecasting"] == "requires_adaptation"
    assert compatibility["optimization"] == "unsupported"
    assert len(compatibility["checks"]) == 8


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "USER_DATASETS", DatasetStore(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), api.APIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    thread.join()
    server.server_close()


def request(port, method, path, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body, headers or {})
    response = conn.getresponse()
    result = response.status, json.loads(response.read())
    conn.close()
    return result


def test_real_multipart_mapping_and_dataset_switching(server):
    boundary = "energy-test-boundary"
    body = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="power.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode() + VALID + f'\r\n--{boundary}--\r\n'.encode()
    status, meta = request(server, "POST", "/api/datasets/upload", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 201
    dataset_id = meta["id"]
    status, _ = request(server, "POST", f"/api/datasets/{dataset_id}/mapping", json.dumps({"mapping": MAPPING, "confirm_conversions": True}), {"Content-Type": "application/json"})
    assert status == 200
    status, data = request(server, "GET", f"/api/datasets/{dataset_id}/analysis")
    assert status == 200
    assert data["summary"]["energy_kwh"]["demand_kw"] == 35
    assert len(data["records"]) == 3
    assert "economic_kpis" not in data
    _, listing = request(server, "GET", "/api/datasets")
    assert listing["datasets"][0]["status"] == "ready"
    assert "root" not in listing["storage"]


def test_http_invalid_payload_and_cross_site_access(server):
    assert request(server, "POST", "/api/datasets/upload", b"invalid", {"Content-Type": "text/csv"})[0] == 400
    assert request(server, "GET", "/api/datasets", headers={"Origin": "https://unrelated.example"})[0] == 403
    assert request(server, "GET", "/api/datasets", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request(server, "POST", "/api/datasets/upload", b"", {"Content-Length": str(MAX_BYTES + 65537)})[0] == 413
    assert request(server, "GET", "/api/datasets/../../configs")[0] == 404


def test_model_and_research_inputs_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "results/part8_comparative_benchmark/test_evaluation").rglob("*.json")) + [root / "configs/forecasting_model.yaml"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    store = DatasetStore(tmp_path)
    meta = store.upload(VALID, "sample.csv")
    store.apply_mapping(meta["id"], {"mapping": MAPPING, "confirm_conversions": True})
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
