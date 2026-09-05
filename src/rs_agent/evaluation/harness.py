from __future__ import annotations

import csv
import html
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import rasterio
from scipy.ndimage import label

from rs_agent.domain import Job, JobRequest, JobStatus, Stage
from rs_agent.evaluation.models import EvaluationCase
from rs_agent.orchestration import WorkflowEngine
from rs_agent.planning import Planner
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore


@dataclass
class CaseResult:
    case_id: str
    route: str
    status: str
    interpret_tool: str | None
    route_correct: bool
    plan_legal: bool
    iou: float | None
    f1: float | None
    area_error_fraction: float | None
    object_count_error: int | None
    qa_passed: bool | None
    repeated_completed_steps: int
    trace_complete: bool
    error: str | None = None
    elapsed_seconds: float = 0
    llm_requests: int = 0
    llm_http_successes: int = 0
    llm_selected_stages: int = 0
    llm_deferred_stages: int = 0
    fallback_stages: int = 0
    total_tokens: int = 0
    evidence_dir: str | None = None
    valid_pixels: int = 0
    sensor_type: str = "unknown"
    resolution_m: float | None = None


class EvaluationHarness:
    """Runs immutable labelled cases and emits comparable rule/LLM evidence."""

    def __init__(self, workspace: str | Path, llm_planner: Planner | None = None) -> None:
        self.workspace = Path(workspace)
        self.llm_planner = llm_planner or Planner()
        self.run_id = uuid4().hex

    @staticmethod
    def load_cases(manifest_dir: str | Path, include_smoke: bool = False) -> list[EvaluationCase]:
        cases: list[EvaluationCase] = []
        for path in sorted(Path(manifest_dir).glob("*.json")):
            case = EvaluationCase.model_validate_json(
                path.read_text(encoding="utf-8")
            ).resolve_paths(path)
            case.validate_files()
            if include_smoke or case.split == "evaluation":
                cases.append(case)
        if not cases:
            raise ValueError("No evaluation manifests found")
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation case IDs must be unique")
        return cases

    def run(self, cases: list[EvaluationCase], route: Literal["rule", "llm"]) -> list[CaseResult]:
        if route == "llm" and self.llm_planner.client is None:
            raise ValueError(
                "LLM evaluation requires a configured client; rule fallback is not an LLM run"
            )
        results = []
        for case in cases:
            case.validate_files()
            started = time.perf_counter()
            planner = Planner() if route == "rule" else self.llm_planner
            call_start = len(getattr(planner.client, "calls", []))
            run_root = self.workspace / uuid4().hex / route / case.case_id
            engine = WorkflowEngine(
                SQLiteStateStore(run_root / "state.sqlite3"),
                LocalArtifactStore(run_root / "artifacts"),
                planner=planner,
            )
            job = engine.submit(
                Job(
                    request=JobRequest(
                        task_type=case.task_type,
                        sensor_type=case.sensor_type,
                        image_uri=case.image_uri,
                        band_indices=case.bands,
                    )
                )
            )
            execution_error = None
            try:
                completed = engine.run(job.job_id)
            except Exception as exc:  # noqa: BLE001 -- isolate a failed case; persist error
                completed = job
                completed.status = JobStatus.FAILED
                execution_error = f"{type(exc).__name__}: {exc}"
            traces = engine.state.traces(job.job_id)
            try:
                result = self._score(case, route, completed, planner, traces)
            except Exception as exc:  # noqa: BLE001 -- scoring errors must not abort the batch
                result = CaseResult(
                    case.case_id,
                    route,
                    "failed",
                    None,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    False,
                    f"scoring error: {type(exc).__name__}: {exc}",
                )
            calls = getattr(planner.client, "calls", [])[call_start:]
            decisions = [
                t["payload"].get("planner", {}).get("source")
                for t in traces
                if t["event_type"] == "plan_selected"
            ]
            result.elapsed_seconds = time.perf_counter() - started
            result.sensor_type = case.sensor_type
            result.resolution_m = case.resolution
            result.llm_requests = len(calls)
            result.llm_http_successes = sum(c.get("http_status") == 200 for c in calls)
            result.total_tokens = sum(c.get("usage", {}).get("total_tokens", 0) for c in calls)
            result.llm_selected_stages = sum(d in ("llm", "llm_repaired") for d in decisions)
            result.llm_deferred_stages = decisions.count("rule_single_eligible_route")
            result.fallback_stages = decisions.count("rule_fallback")
            result.evidence_dir = str(run_root.resolve())
            result.error = execution_error or result.error
            for filename, payload in (
                ("case.json", case.model_dump(mode="json")),
                ("job.json", completed.model_dump(mode="json")),
                ("traces.json", traces),
                ("llm_calls.json", calls),
                ("result.json", asdict(result)),
            ):
                (run_root / filename).write_text(
                    json.dumps(payload, indent=2, default=str), encoding="utf-8"
                )
            results.append(result)
            print(
                f"{route} {case.case_id}: {result.status}; IoU={result.iou}; HTTP200={result.llm_http_successes}",
                flush=True,
            )
        return results

    def run_comparison(self, cases: list[EvaluationCase]) -> dict[str, list[CaseResult]]:
        return {"rule": self.run(cases, "rule"), "llm": self.run(cases, "llm")}

    def write_reports(
        self, comparison: dict[str, list[CaseResult]], output_dir: str | Path
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = [asdict(item) for results in comparison.values() for item in results]
        by_sensor = {
            sensor: self.summary(
                {
                    route: [r for r in items if r.sensor_type == sensor]
                    for route, items in comparison.items()
                }
            )
            for sensor in ("optical", "sar")
        }
        json_path = output / "evaluation.json"
        json_path.write_text(
            json.dumps(
                {"summary": self.summary(comparison), "by_sensor": by_sensor, "cases": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
        csv_path = output / "evaluation.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0]) if rows else list(CaseResult.__annotations__)
            )
            writer.writeheader()
            writer.writerows(rows)
        html_path = output / "evaluation.html"
        html_path.write_text(
            self._html({"overall": self.summary(comparison), "by_sensor": by_sensor}, rows),
            encoding="utf-8",
        )
        return {"json": json_path, "csv": csv_path, "html": html_path}

    @staticmethod
    def _score(
        case: EvaluationCase,
        route: str,
        job: Job,
        planner: Planner,
        traces: list[dict[str, object]],
    ) -> CaseResult:
        plan = job.plans.get("interpret")
        completed_attempts = [attempt for attempt in job.attempts if attempt.status == "completed"]
        repeated = len(completed_attempts) - len(
            {attempt.step_id for attempt in completed_attempts}
        )
        trace_events = {
            "job_submitted",
            "plan_selected",
            "tool_started",
            "tool_completed",
            "state_transition",
        }
        if job.status not in (JobStatus.COMPLETED, JobStatus.WAITING_HUMAN):
            return CaseResult(
                case.case_id,
                route,
                job.status,
                plan.tool if plan else None,
                False,
                False,
                None,
                None,
                None,
                None,
                None,
                repeated,
                False,
                "workflow did not complete",
            )
        mask_kind = "water_mask_geotiff" if case.task_type == "water_extraction" else "building_mask_geotiff"
        mask = next((item for item in reversed(job.artifacts) if item.kind == mask_kind), None)
        if mask is None:
            return CaseResult(
                case.case_id,
                route,
                job.status,
                plan.tool if plan else None,
                False,
                False,
                None,
                None,
                None,
                None,
                None,
                repeated,
                False,
                "mask artifact missing",
            )
        with (
            rasterio.open(Path(str(mask.metadata["path"]))) as pred,
            rasterio.open(case.label_uri) as truth,
        ):
            if (
                pred.shape != truth.shape
                or pred.crs != truth.crs
                or not pred.transform.almost_equals(truth.transform)
            ):
                raise ValueError(f"Label grid differs from result for {case.case_id}")
            raw = truth.read(1)
            valid = (truth.read_masks(1) > 0) & np.isin(raw, [0, 1])
            if not valid.any():
                raise ValueError("No valid labelled pixels")
            predicted = (pred.read(1) == 1) & valid
            expected = (raw == 1) & valid
        intersection = int(np.logical_and(predicted, expected).sum())
        union = int(np.logical_or(predicted, expected).sum())
        predicted_count, expected_count = int(predicted.sum()), int(expected.sum())
        precision_recall_denom = predicted_count + expected_count
        plan_legal = False
        if plan:
            try:
                for stage_name, selected in job.plans.items():
                    planner.registry.validate(
                        Stage(stage_name), case.sensor_type, case.task_type, selected
                    )
                plan_legal = all(stage.value in job.plans for stage in Planner.PLANNED_STAGES)
            except ValueError:
                pass
        return CaseResult(
            case_id=case.case_id,
            route=route,
            status=job.status,
            interpret_tool=plan.tool if plan else None,
            route_correct=bool(plan and plan.tool == case.expected_interpret_tool),
            plan_legal=plan_legal,
            iou=intersection / union if union else 1.0,
            f1=(2 * intersection / precision_recall_denom) if precision_recall_denom else 1.0,
            area_error_fraction=abs(predicted_count - expected_count) / expected_count
            if expected_count
            else None,
            object_count_error=abs(
                EvaluationHarness._objects(predicted) - EvaluationHarness._objects(expected)
            ),
            qa_passed=job.quality.passed if job.quality else None,
            repeated_completed_steps=repeated,
            trace_complete=trace_events.issubset({str(item["event_type"]) for item in traces}),
            valid_pixels=int(valid.sum()),
        )

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        with rasterio.open(path) as dataset:
            return dataset.read(1).astype(bool)

    @staticmethod
    def _objects(mask: np.ndarray) -> int:
        return int(label(mask)[1])

    @staticmethod
    def summary(
        comparison: dict[str, list[CaseResult]],
    ) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for route, rows in comparison.items():
            completed = [row for row in rows if row.status == "completed"]
            result[route] = {
                "cases": len(rows),
                "success_rate": len(completed) / len(rows) if rows else 0.0,
                "route_accuracy": mean([float(row.route_correct) for row in rows]) if rows else 0.0,
                "plan_legality": mean([float(row.plan_legal) for row in rows]) if rows else 0.0,
                "mean_iou": mean([row.iou for row in rows if row.iou is not None])
                if any(row.iou is not None for row in rows)
                else None,
                "mean_f1": mean([row.f1 for row in rows if row.f1 is not None])
                if any(row.f1 is not None for row in rows)
                else None,
                "repeat_steps": sum(row.repeated_completed_steps for row in rows),
                "llm_requests": sum(row.llm_requests for row in rows),
                "llm_http_successes": sum(row.llm_http_successes for row in rows),
                "llm_selected_stages": sum(row.llm_selected_stages for row in rows),
                "llm_deferred_stages": sum(row.llm_deferred_stages for row in rows),
                "fallback_stages": sum(row.fallback_stages for row in rows),
                "total_tokens": sum(row.total_tokens for row in rows),
                "mean_elapsed_seconds": mean([row.elapsed_seconds for row in rows])
                if rows
                else None,
                "trace_completeness": mean([float(row.trace_complete) for row in rows])
                if rows
                else None,
                "human_intervention_rate": mean(
                    [float(row.status == "waiting_human") for row in rows]
                )
                if rows
                else 0.0,
            }
        return result

    @staticmethod
    def _html(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        headings = list(rows[0]) if rows else []
        cells = "".join(f"<th>{html.escape(item)}</th>" for item in headings)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headings) + "</tr>"
            for row in rows
        )
        return f"<!doctype html><meta charset=utf-8><title>RS Agent evaluation</title><h1>Evaluation report</h1><pre>{html.escape(json.dumps(summary, indent=2))}</pre><table border=1><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table>"
