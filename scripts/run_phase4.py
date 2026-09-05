"""One command: verified fixed corpus -> rule baseline + real LLM -> CSV/JSON/HTML.

Use --prepare to acquire missing inputs, --rule-only for offline checks.
Credentials are loaded explicitly from project .env and never copied to evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from dotenv import dotenv_values

from rs_agent.evaluation import EvaluationHarness
from rs_agent.planning import Planner
from rs_agent.planning.planner import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        from prepare_phase4 import main as prepare

        prepare()
    cases = EvaluationHarness.load_cases(
        ROOT / "data/manifests", include_smoke=args.smoke, task_type="water_extraction"
    )
    if args.smoke:
        cases = [c for c in cases if c.split == "smoke"]
        assert len(cases) == 4
    else:
        assert len(cases) == 24 and sum(c.sensor_type == "optical" for c in cases) == 12
        assert {c.resolution for c in cases} == {10, 20, 30}
    config = dotenv_values(ROOT / ".env")
    if not args.rule_only and not all(
        config.get(k) for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL")
    ):
        raise RuntimeError(
            "Project .env is missing model configuration; no LLM run will be claimed"
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "evaluation-output" / ("smoke-" if args.smoke else "phase4-") / stamp
    output.mkdir(parents=True)
    fingerprints = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "data/manifests").glob("*.json"))
    }
    metadata = {
        "started_utc": stamp,
        "manifest_sha256": fingerprints,
        "scope": "water segmentation; paired sensors are not independent scenes",
        "pricing": "currency cost unavailable; actual provider token usage recorded",
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "rasterio", "httpx", "pydantic", "rs-agent")
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    comparison = {}
    reporter = EvaluationHarness(output / "evidence")
    for route in ["rule"] if args.rule_only else ["rule", "llm"]:

        def run_case(case, route=route):
            client = (
                None
                if route == "rule"
                else DeepSeekClient(
                    config["DEEPSEEK_API_KEY"],
                    config["DEEPSEEK_MODEL"],
                    config.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                    timeout_seconds=90,
                )
            )
            return EvaluationHarness(output / "evidence", Planner(client=client)).run(
                [case], route
            )[0]

        comparison[route] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(run_case, cases):
                comparison[route].append(result)
                if route == "rule":
                    baseline_dir = ROOT / "data/baselines"
                    baseline_dir.mkdir(exist_ok=True)
                    baseline_path = baseline_dir / f"{result.case_id}.json"
                    if not baseline_path.exists():
                        baseline_path.write_text(
                            json.dumps(
                                {
                                    "version": "deterministic-water-v2",
                                    "iou": result.iou,
                                    "f1": result.f1,
                                    "area_error_fraction": result.area_error_fraction,
                                    "object_count_error": result.object_count_error,
                                    "status": result.status,
                                    "evidence_dir": result.evidence_dir,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                reporter.write_reports(comparison, output)
    print(json.dumps(reporter.summary(comparison), indent=2), flush=True)
    print(f"REPORT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
