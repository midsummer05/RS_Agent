"""Evaluate the fixed building corpus with a rule baseline and real LLM planning.

Use --rule-only only for offline diagnostics.  The default refuses to run when
the real provider configuration is absent, rather than labelling a fallback as
an LLM evaluation.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from rs_agent.evaluation import EvaluationHarness
from rs_agent.planning import Planner
from rs_agent.planning.planner import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--rule-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        from prepare_building_phase4 import main as prepare

        prepare()
    cases = EvaluationHarness.load_cases(
        ROOT / "data/manifests", task_type="building_extraction"
    )
    if len(cases) != 12 or {case.sensor_type for case in cases} != {"optical"}:
        raise ValueError("Building formal corpus must be exactly 12 optical labelled cases")
    config = dotenv_values(ROOT / ".env")
    if not args.rule_only and not all(config.get(key) for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL")):
        raise RuntimeError("Project .env is missing model configuration; no LLM run will be claimed")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "evaluation-output" / "building-phase4-" / stamp
    harness = EvaluationHarness(output / "evidence")
    comparison = {"rule": harness.run(cases, "rule")}
    if not args.rule_only:

        def run_llm_case(case):
            planner = Planner(
                client=DeepSeekClient(
                    config["DEEPSEEK_API_KEY"],
                    config["DEEPSEEK_MODEL"],
                    config.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                    timeout_seconds=90,
                )
            )
            return EvaluationHarness(output / "evidence", llm_planner=planner).run([case], "llm")[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            comparison["llm"] = list(pool.map(run_llm_case, cases))
    harness.write_reports(comparison, output)
    baseline_dir = ROOT / "data" / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    for result in comparison["rule"]:
        baseline_path = baseline_dir / f"{result.case_id}.json"
        if baseline_path.exists():
            continue
        baseline_path.write_text(
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
    print(json.dumps(harness.summary(comparison), indent=2), flush=True)
    print(f"REPORT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
