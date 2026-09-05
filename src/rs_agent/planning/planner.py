from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Protocol

import httpx
from pydantic import ValidationError

from rs_agent.domain import ExecutionPlan, Job, Stage
from rs_agent.planning.registry import ToolRegistry


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class DeepSeekClient:
    """Small OpenAI-compatible client. It never logs credentials or prompt bodies."""

    def __init__(
        self, api_key: str, model: str, base_url: str, timeout_seconds: float = 30
    ) -> None:
        self.api_key, self.model = api_key, model
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str) -> str:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "model": self.model,
            "endpoint": self.base_url + "/chat/completions",
        }
        self.calls.append(record)
        try:
            return self._complete(system, user, record)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started

    def _complete(self, system: str, user: str, record: dict[str, Any]) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            timeout=self.timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        record["http_status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        record.update(
            request_id=payload.get("id"),
            response_model=payload.get("model"),
            usage=payload.get("usage", {}),
        )
        return str(payload["choices"][0]["message"]["content"])


class RuleRouter:
    def plan(self, stage: Stage, job: Job) -> ExecutionPlan:
        tool = {
            Stage.PREPROCESS: "image_adapter_normalize",
            Stage.INTERPRET: "optical_ndwi"
            if job.request.sensor_type == "optical"
            else "sar_adaptive_threshold",
            Stage.POSTPROCESS: "morphology_and_polygonize",
            Stage.QA: "water_statistics_and_geometry_qa",
            Stage.REPORT: "markdown_report",
        }[stage]
        parameters: dict[str, Any] = {
            "optical_ndwi": {"threshold": 0.0},
            "sar_adaptive_threshold": {"percentile": 35.0},
            "morphology_and_polygonize": {"min_component_pixels": 9.0},
            "water_statistics_and_geometry_qa": {
                "min_coverage_fraction": 0.0,
                "max_coverage_fraction": 0.98,
            },
        }.get(tool, {})
        return ExecutionPlan(tool=tool, parameters=parameters, rationale="deterministic rule route")


class Planner:
    # Phase 2: LLM planning occurs at each mutable execution stage. The report
    # is deterministic because it only formats already-approved statistics.
    PLANNED_STAGES: ClassVar[frozenset[Stage]] = frozenset(
        {Stage.PREPROCESS, Stage.INTERPRET, Stage.POSTPROCESS, Stage.QA}
    )

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        client: LLMClient | None = None,
        rule_router: RuleRouter | None = None,
    ) -> None:
        self.registry, self.client, self.rule_router = (
            registry or ToolRegistry(),
            client,
            rule_router or RuleRouter(),
        )

    def plan(
        self, stage: Stage, job: Job, experience_memory: list[dict[str, object]] | None = None
    ) -> ExecutionPlan:
        fallback = self.rule_router.plan(stage, job)
        if self.client is None or stage not in self.PLANNED_STAGES:
            self.last_decision = {
                "source": "rule_fixed_stage"
                if stage not in self.PLANNED_STAGES
                else "rule_no_llm_client"
            }
            return fallback
        context = self._context(stage, job, experience_memory)
        candidate, error = self._ask(context, repair_error=None)
        if candidate is not None:
            return self._accept_candidate(candidate, fallback)
        repaired, _ = self._ask(context, repair_error=error)
        if repaired is not None:
            return self._accept_candidate(repaired, fallback, repaired=True)
        self.last_decision = {"source": "rule_fallback", "reason": error}
        return fallback

    def _accept_candidate(
        self, candidate: ExecutionPlan, fallback: ExecutionPlan, repaired: bool = False
    ) -> ExecutionPlan:
        """Attach a deterministic, compatible fallback to every LLM plan."""
        candidate = candidate.model_copy(update={"fallback_tool": fallback.tool})
        self.last_decision = {"source": "llm_repaired" if repaired else "llm"}
        return candidate

    def context(
        self, stage: Stage, job: Job, experience_memory: list[dict[str, object]] | None = None
    ) -> dict[str, Any]:
        """A trace-safe planning summary; it intentionally has no image bytes or arrays."""
        return self._context(stage, job, experience_memory)

    def _ask(
        self, context: dict[str, Any], repair_error: str | None
    ) -> tuple[ExecutionPlan | None, str]:
        system = (
            "You are a remote-sensing stage planner. Return only a JSON object conforming "
            "to the supplied schema. Choose only an allowed tool; do not request image pixels. "
            "Choose parameters exactly from the registered finite parameter profiles."
        )
        message: dict[str, Any] = {
            "context": context,
            "execution_plan_schema": ExecutionPlan.model_json_schema(),
        }
        if repair_error:
            message["repair_instruction"] = (
                f"Previous plan was rejected: {repair_error}. Return a corrected plan."
            )
        try:
            raw = self.client.complete(system, json.dumps(message, ensure_ascii=False))  # type: ignore[union-attr]
            plan = ExecutionPlan.model_validate_json(raw)
            self.registry.validate_profile(
                Stage(context["stage"]), context["sensor_type"], context["task_type"], plan
            )
            return plan, ""
        except (
            httpx.HTTPError,
            KeyError,
            json.JSONDecodeError,
            TypeError,
            ValidationError,
            ValueError,
        ) as exc:
            return None, str(exc)[:500]

    def _context(
        self, stage: Stage, job: Job, experience_memory: list[dict[str, object]] | None = None
    ) -> dict[str, Any]:
        return {
            "stage": stage.value,
            "task_type": job.request.task_type,
            "sensor_type": job.request.sensor_type,
            "image_metadata": job.image_manifest,
            "artifacts": [
                {"kind": item.kind, "uri": item.uri, "sha256": item.sha256}
                for item in job.artifacts
            ],
            "prior_errors": [item.error for item in job.attempts if item.error],
            "task_memory_refs": job.memory_refs,
            "experience_memory": experience_memory or [],
            "allowed_tools": self.registry.context_contracts(
                stage, job.request.sensor_type, job.request.task_type
            ),
            "routing_policy": {
                "decision_type": "tool_and_registered_parameter_profile",
                "default_plan": self.rule_router.plan(stage, job).model_dump(mode="json"),
                "registered_parameter_profiles": self.registry.profiles_for(
                    stage, job.request.sensor_type, job.request.task_type
                ),
                "parameter_policy": "Use exact registered profile values only; no free-form numeric tuning.",
                "image_diagnostics_note": "Summary statistics only; no image pixels or labels are exposed.",
            },
        }
