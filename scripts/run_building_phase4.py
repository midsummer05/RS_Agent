"""Run the fixed twelve-case building corpus without silently using an LLM."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rs_agent.evaluation import EvaluationHarness

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        from prepare_building_phase4 import main as prepare

        prepare()
    cases = EvaluationHarness.load_cases(
        ROOT / "data/manifests", task_type="building_extraction"
    )
    if len(cases) != 12 or {case.sensor_type for case in cases} != {"optical"}:
        raise ValueError("Building formal corpus must be exactly 12 optical labelled cases")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "evaluation-output" / "building-phase4-" / stamp
    harness = EvaluationHarness(output / "evidence")
    results = harness.run(cases, "rule")
    harness.write_reports({"rule": results}, output)
    baseline_dir = ROOT / "data" / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    for result in results:
        (baseline_dir / f"{result.case_id}.json").write_text(
            json.dumps(
                {
                    "version": "deterministic-building-rgb-v1",
                    "iou": result.iou,
                    "f1": result.f1,
                    "area_error_fraction": result.area_error_fraction,
                    "object_count_error": result.object_count_error,
                    "status": result.status,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(harness.summary({"rule": results}), indent=2), flush=True)
    print(f"REPORT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
