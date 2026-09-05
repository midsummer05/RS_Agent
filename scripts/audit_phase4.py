"""Read-only corpus/evidence validation. Writes an audit report beside a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import rasterio
from dotenv import dotenv_values

from rs_agent.evaluation import EvaluationHarness

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.report_dir / "evaluation.json").read_text())
    metadata = json.loads((args.report_dir / "run_metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Source changed since execution: {name}"
    cases = EvaluationHarness.load_cases(ROOT / "data/manifests", include_smoke=True)
    counts = Counter(f"{c.split}_{c.sensor_type}" for c in cases)
    assert counts == {
        "evaluation_optical": 12,
        "evaluation_sar": 12,
        "smoke_optical": 2,
        "smoke_sar": 2,
    }, counts
    evaluation_ids = {c.case_id for c in cases if c.split == "evaluation"}
    eval_chips = {c.provenance["chip"] for c in cases if c.split == "evaluation"}
    smoke_chips = {c.provenance["chip"] for c in cases if c.split == "smoke"}
    assert not eval_chips & smoke_chips
    for case in cases:
        for source in case.provenance["source_files"].values():
            relative = source["url"].split("/v1.1/", 1)[1]
            assert sha(ROOT / "data/sources/sen1floods11-v1.1" / relative) == source["sha256"]
        with rasterio.open(case.image_uri) as image, rasterio.open(case.label_uri) as label:
            assert image.crs == label.crs and image.transform == label.transform
            assert image.shape == label.shape
        assert (
            sha(ROOT / "data/manifests" / f"{case.case_id}.json")
            == metadata["manifest_sha256"][f"{case.case_id}.json"]
        )
        if case.split == "evaluation":
            assert (ROOT / "data/manifests" / case.baseline_metrics["reference"]).is_file()
    assert len(report["cases"]) == 48
    for route in ("rule", "llm"):
        assert {r["case_id"] for r in report["cases"] if r["route"] == route} == evaluation_ids
    request_ids = []
    artifact_count = 0
    secret = (dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY") or "").encode()
    if secret:
        for path in args.report_dir.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".html", ".sqlite3", ".md"):
                assert secret not in path.read_bytes(), "Credential found in persisted evidence"
    for row in report["cases"]:
        evidence = Path(row["evidence_dir"])
        assert (evidence / "state.sqlite3").is_file()
        job = json.loads((evidence / "job.json").read_text())
        assert json.loads((evidence / "result.json").read_text()) == row
        traces = json.loads((evidence / "traces.json").read_text())
        assert traces
        calls = json.loads((evidence / "llm_calls.json").read_text())
        assert len(calls) == row["llm_requests"]
        if row["route"] == "llm":
            if row["llm_selected_stages"]:
                assert row["llm_http_successes"] >= row["llm_selected_stages"]
                request_ids.extend(c["request_id"] for c in calls if c.get("http_status") == 200)
            else:
                assert row["llm_deferred_stages"] >= 1 and not calls
        else:
            assert not calls
        for artifact in job["artifacts"]:
            assert sha(Path(artifact["metadata"]["path"])) == artifact["sha256"]
            artifact_count += 1
    assert len(request_ids) == len(set(request_ids)), "Repeated provider request IDs"
    result = {
        "verified": True,
        "counts": dict(counts),
        "evaluation_scene_pairs": len(eval_chips),
        "smoke_disjoint": True,
        "task_evidence_bundles": 48,
        "provider_unique_successful_requests": len(request_ids),
        "verified_artifacts": artifact_count,
        "credential_scan_passed": bool(secret),
        "summary": report["summary"],
    }
    (args.report_dir / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
