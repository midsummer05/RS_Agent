from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from rs_agent.domain import (
    STAGES,
    ApprovalAction,
    ApprovalEvent,
    Attempt,
    ExecutionPlan,
    FailureInjection,
    Job,
    JobStatus,
    QualityResult,
    Stage,
)
from rs_agent.planning import Planner
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore
from rs_agent.tools import DeterministicToolchain, MockToolRegistry


class WorkflowEngine:
    def __init__(
        self,
        state: SQLiteStateStore,
        artifacts: LocalArtifactStore,
        tools: MockToolRegistry | None = None,
        deterministic_tools: DeterministicToolchain | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.state, self.artifacts = state, artifacts
        self.tools = tools or MockToolRegistry()
        self.deterministic_tools = deterministic_tools or DeterministicToolchain()
        self.planner = planner or Planner()

    def submit(self, job: Job) -> Job:
        self.state.save(job, "submitted")
        self.state.trace(
            job.job_id,
            "job_submitted",
            {"task_type": job.request.task_type, "sensor_type": job.request.sensor_type},
        )
        return job

    def _trace(self, job: Job, event_type: str, payload: dict[str, object]) -> None:
        self.state.trace(job.job_id, event_type, payload)

    @staticmethod
    def _injected(job: Job, failure: FailureInjection, stage: Stage) -> bool:
        key = f"{failure.value}:{stage.value}"
        if (
            failure not in job.request.failure_injections
            or job.request.injection_stage is not stage
            or key in job.injected_failure_keys
        ):
            return False
        job.injected_failure_keys.append(key)
        return True

    @staticmethod
    def _validate_next_stage(job: Job, stage: Stage) -> None:
        """Reject corrupt checkpoints that would skip or reorder the fixed lifecycle."""
        expected = (
            STAGES[len(job.completed_stages)] if len(job.completed_stages) < len(STAGES) else None
        )
        if expected is not stage:
            raise ValueError(f"Invalid stage transition: expected {expected}, got {stage}")

    def run(self, job_id: UUID | str) -> Job:
        job = self.state.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        if job.status is JobStatus.COMPLETED:
            return job
        job.status = JobStatus.RUNNING
        self.state.save(job, "run_started")
        self._trace(job, "state_transition", {"status": job.status, "reason": "run_started"})
        for stage in STAGES:
            if stage in job.completed_stages:
                continue
            self._validate_next_stage(job, stage)
            job.stage = stage
            is_mock = job.request.image_uri.startswith("mock://")
            is_manual_edit = bool(
                job.approval
                and job.approval.action is ApprovalAction.EDIT
                and job.approval.stage is stage
                and stage.value in job.plans
            )
            if is_manual_edit:
                plan = job.plans[stage.value]
                self._trace(
                    job,
                    "human_plan_applied",
                    {"stage": stage, "tool": plan.tool, "parameters": plan.parameters},
                )
            elif is_mock:
                plan = self.tools.plan(stage)
            elif self._injected(job, FailureInjection.INVALID_PLANNER_OUTPUT, stage):
                injected = ExecutionPlan(tool="injected_invalid_tool", parameters={})
                try:
                    self.planner.registry.validate(
                        stage, job.request.sensor_type, job.request.task_type, injected
                    )
                except ValueError as exc:
                    self._trace(
                        job,
                        "planner_invalid_output",
                        {"stage": stage, "error": str(exc), "recovered_by": "rule_router"},
                    )
                plan = self.planner.rule_router.plan(stage, job)
            else:
                experience = self.state.memories(job.request.sensor_type, job.request.task_type)
                self._trace(job, "planner_context", self.planner.context(stage, job, experience))
                plan = self.planner.plan(stage, job, experience)
            job.plans[stage.value] = plan
            self._trace(
                job,
                "plan_selected",
                {
                    "stage": stage,
                    "tool": plan.tool,
                    "parameters": plan.parameters,
                    "rationale": plan.rationale,
                    "planner": self.planner.last_decision if not is_mock else {"source": "mock"},
                },
            )
            attempt = Attempt(
                step_id=f"{job.job_id}:{stage.value}",
                stage=stage,
                tool=plan.tool,
                idempotency_key=f"{job.job_id}:{stage.value}:{plan.version}",
                status="running",
            )
            job.attempts.append(attempt)
            self.state.save(job, "stage_started")
            self._trace(
                job,
                "tool_started",
                {
                    "stage": stage,
                    "tool": plan.tool,
                    "parameters": plan.parameters,
                    "step_id": attempt.step_id,
                },
            )
            if job.request.fail_once_stage is stage and stage not in job.injected_failures:
                job.injected_failures.append(stage)
                attempt.status, attempt.error = "failed", "injected recoverable failure"
                attempt.finished_at = datetime.now(UTC)
                job.status = JobStatus.FAILED
                self.state.save(job, "recoverable_error")
                self._trace(
                    job,
                    "tool_failed",
                    {"stage": stage, "error": attempt.error, "recoverable": True},
                )
                return job
            if self._injected(job, FailureInjection.TIMEOUT, stage):
                attempt.status, attempt.error, attempt.finished_at = (
                    "failed",
                    "injected timeout",
                    datetime.now(UTC),
                )
                job.status = JobStatus.FAILED
                self.state.save(job, "recoverable_timeout")
                self._trace(
                    job,
                    "tool_failed",
                    {"stage": stage, "error": attempt.error, "recoverable": True},
                )
                return job
            try:
                if self._injected(job, FailureInjection.FILE_MISSING, stage):
                    raise FileNotFoundError("injected missing artifact")
                if is_mock:
                    output = self.tools.run(stage, str(job.job_id))
                    job.artifacts.append(self.artifacts.put_json(f"{stage.value}_result", output))
                    if stage is Stage.QA:
                        job.quality = QualityResult(
                            passed=True, metrics={"mock_quality_score": output["quality_score"]}
                        )
                else:
                    # Validate once more immediately before a real tool boundary.
                    self.planner.registry.validate(
                        stage, job.request.sensor_type, job.request.task_type, plan
                    )
                    output = self.deterministic_tools.run(stage, job, self.artifacts, plan)
                    job.artifacts.extend(output.artifacts)
                    if output.quality:
                        job.quality = output.quality
                if stage is Stage.QA and self._injected(job, FailureInjection.QA_FAILURE, stage):
                    job.quality = QualityResult(
                        passed=False,
                        metrics=job.quality.metrics if job.quality else {},
                        notes=["injected QA failure"],
                    )
            except (FileNotFoundError, ValueError, OSError) as exc:
                attempt.status, attempt.error = "failed", str(exc)
                attempt.finished_at = datetime.now(UTC)
                job.status = JobStatus.FAILED
                self.state.save(job, "tool_error")
                self._trace(
                    job, "tool_failed", {"stage": stage, "error": str(exc), "recoverable": False}
                )
                return job
            attempt.status, attempt.finished_at = "completed", datetime.now(UTC)
            job.completed_stages.append(stage)
            self.state.save(job, "stage_completed")
            self._trace(
                job,
                "tool_completed",
                {
                    "stage": stage,
                    "tool": plan.tool,
                    "duration_seconds": (attempt.finished_at - attempt.started_at).total_seconds(),
                    "artifacts": [item.uri for item in job.artifacts[-len(output.artifacts) :]]
                    if not is_mock
                    else [],
                },
            )
            if stage is Stage.QA and job.quality and not job.quality.passed:
                job.status = JobStatus.WAITING_HUMAN
                self.state.save(job, "qa_waiting_human")
                self._trace(
                    job,
                    "state_transition",
                    {
                        "status": job.status,
                        "reason": "quality_not_approved",
                        "quality": job.quality.model_dump(mode="json"),
                    },
                )
                return job
        self._record_experience(job)
        job.status = JobStatus.COMPLETED
        self.state.save(job, "completed")
        self._trace(job, "state_transition", {"status": job.status, "reason": "completed"})
        return job

    def resume(self, job_id: UUID | str) -> Job:
        job = self.state.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        if job.status not in {JobStatus.FAILED, JobStatus.PENDING, JobStatus.RUNNING}:
            return job
        return self.run(job_id)

    def apply_approval(
        self,
        job_id: UUID | str,
        action: ApprovalAction,
        actor: str,
        comment: str | None = None,
        stage: Stage | None = None,
        parameters: dict[str, float] | None = None,
    ) -> Job:
        job = self.state.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        if job.status is not JobStatus.WAITING_HUMAN:
            raise ValueError("Job is not waiting for human approval")
        parameters = parameters or {}
        restart_stage = stage or Stage.INTERPRET
        if action is ApprovalAction.APPROVE:
            job.status = JobStatus.PENDING
        else:
            index = STAGES.index(restart_stage)
            job.completed_stages = [
                item for item in job.completed_stages if STAGES.index(item) < index
            ]
            for item in STAGES[index:]:
                job.plans.pop(item.value, None)
            if action is ApprovalAction.EDIT:
                plan = self.planner.rule_router.plan(restart_stage, job)
                plan.parameters.update(parameters)
                self.planner.registry.validate(
                    restart_stage, job.request.sensor_type, job.request.task_type, plan
                )
                job.plans[restart_stage.value] = plan
            job.stage, job.status = restart_stage, JobStatus.PENDING
        job.approval = ApprovalEvent(
            action=action, actor=actor, comment=comment, stage=stage, parameters=parameters
        )
        self.state.save(job, f"human_{action.value}")
        self._trace(job, "human_approval", job.approval.model_dump(mode="json"))
        return job

    def _record_experience(self, job: Job) -> None:
        # Long-term memory contains only reusable parameter choices and quality summary.
        for stage, plan in job.plans.items():
            reference = self.state.remember(
                job.request.sensor_type,
                job.request.task_type,
                stage,
                {
                    "tool": plan.tool,
                    "parameters": plan.parameters,
                    "quality_passed": job.quality.passed if job.quality else None,
                },
            )
            if reference not in job.memory_refs:
                job.memory_refs.append(reference)
