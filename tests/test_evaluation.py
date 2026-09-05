import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from rs_agent.evaluation import EvaluationHarness


def test_llm_route_rejects_missing_client(tmp_path):
    with pytest.raises(ValueError, match="requires a configured client"):
        EvaluationHarness(tmp_path).run([], "llm")


def test_evaluation_harness_runs_both_routes_and_writes_reports(tmp_path):
    image = tmp_path / "image.tif"
    label = tmp_path / "label.tif"
    pixels = np.full((4, 20, 20), 100, dtype="uint16")
    pixels[1, :10, :] = 900
    pixels[3, :10, :] = 50
    expected = np.zeros((20, 20), dtype="uint8")
    expected[:10, :] = 1
    profile = {
        "driver": "GTiff",
        "height": 20,
        "width": 20,
        "crs": "EPSG:3857",
        "transform": from_origin(0, 200, 10, 10),
    }
    with rasterio.open(image, "w", count=4, dtype="uint16", **profile) as dataset:
        dataset.write(pixels)
    with rasterio.open(label, "w", count=1, dtype="uint8", **profile) as dataset:
        dataset.write(expected, 1)
    manifest = {
        "case_id": "smoke_optical_01",
        "split": "smoke",
        "sensor_type": "optical",
        "image_uri": "image.tif",
        "label_uri": "label.tif",
        "source": {
            "name": "synthetic test fixture",
            "url": "https://example.invalid",
            "license": "CC0",
        },
        "bands": {"green": 1, "nir": 3},
        "expected_interpret_tool": "optical_ndwi",
    }
    manifest_path = tmp_path / "case.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    harness = EvaluationHarness(tmp_path / "runtime")
    cases = harness.load_cases(tmp_path, include_smoke=True)
    comparison = {"rule": harness.run(cases, "rule")}
    reports = harness.write_reports(comparison, tmp_path / "reports")
    assert comparison["rule"][0].status == "completed"
    assert comparison["rule"][0].route_correct
    assert comparison["rule"][0].trace_complete
    assert all(path.is_file() for path in reports.values())
    evidence = comparison["rule"][0].evidence_dir
    from pathlib import Path

    assert (Path(evidence) / "traces.json").is_file()
    assert comparison["rule"][0].llm_requests == 0
    # Ignored truth pixels are excluded, not converted to water by bool(-1).
    with rasterio.open(label, "w", count=1, dtype="int16", nodata=-1, **profile) as dataset:
        ignored = expected.astype("int16")
        ignored[:5, :] = -1
        dataset.write(ignored, 1)
    rerun = harness.run(cases, "rule")[0]
    assert rerun.valid_pixels == 300
    with rasterio.open(label, "r+") as dataset:
        dataset.transform = from_origin(10, 200, 10, 10)
    assert "grid differs" in harness.run(cases, "rule")[0].error


def test_real_client_records_response_metadata_without_credentials(monkeypatch):
    import httpx

    from rs_agent.planning.planner import DeepSeekClient

    def reply(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "https://provider.invalid/chat/completions"),
            json={
                "id": "test-request-id",
                "model": "test-model",
                "usage": {"total_tokens": 12},
                "choices": [{"message": {"content": '{"tool":"x"}'}}],
            },
        )

    monkeypatch.setattr(httpx, "post", reply)
    client = DeepSeekClient("secret-do-not-persist", "test-model", "https://provider.invalid")
    client.complete("system", "private-prompt")
    assert client.calls[0]["request_id"] == "test-request-id"
    assert client.calls[0]["usage"]["total_tokens"] == 12
    assert "secret-do-not-persist" not in json.dumps(client.calls)
    assert "private-prompt" not in json.dumps(client.calls)
