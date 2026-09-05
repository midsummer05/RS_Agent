from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rs_agent.api.main import engine


def main() -> None:
    parser = argparse.ArgumentParser(prog="rs-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    resume = sub.add_parser("resume", help="resume a checkpointed job")
    resume.add_argument("job_id")
    approval = sub.add_parser("approval", help="record an approval decision and queue the job")
    approval.add_argument("job_id")
    approval.add_argument("action", choices=["approve", "edit", "reject"])
    approval.add_argument("--actor", default="cli-user")
    approval.add_argument("--comment")
    approval.add_argument(
        "--stage", choices=["preprocess", "interpret", "postprocess", "qa", "report"]
    )
    approval.add_argument("--parameters", default="{}", help="JSON parameter overrides for edit")
    worker = sub.add_parser("worker", help="poll and execute queued jobs")
    worker.add_argument("--poll-seconds", type=float, default=2)
    evaluate = sub.add_parser("evaluate", help="run versioned labelled evaluation manifests")
    evaluate.add_argument("manifest_dir")
    evaluate.add_argument("--output", default="evaluation-output")
    evaluate.add_argument("--workspace", default="evaluation-runtime")
    evaluate.add_argument("--include-smoke", action="store_true")
    evaluate.add_argument("--route", choices=["rule", "llm", "both"], default="both")
    args = parser.parse_args()
    if args.command == "resume":
        print(engine().resume(args.job_id).model_dump_json(indent=2))
        return
    if args.command == "approval":
        from rs_agent.domain import ApprovalAction, Stage

        stage = Stage(args.stage) if args.stage else None
        result = engine().apply_approval(
            args.job_id,
            ApprovalAction(args.action),
            args.actor,
            args.comment,
            stage,
            json.loads(args.parameters),
        )
        print(result.model_dump_json(indent=2))
        return
    if args.command == "evaluate":
        from rs_agent.evaluation import EvaluationHarness

        harness = EvaluationHarness(Path(args.workspace), llm_planner=engine().planner)
        cases = harness.load_cases(args.manifest_dir, include_smoke=args.include_smoke)
        comparison = (
            harness.run_comparison(cases)
            if args.route == "both"
            else {args.route: harness.run(cases, args.route)}
        )
        reports = harness.write_reports(comparison, args.output)
        print(json.dumps({key: str(value) for key, value in reports.items()}, indent=2))
        return
    while True:
        for job in engine().state.queued():
            engine().resume(job.job_id)
        time.sleep(args.poll_seconds)
