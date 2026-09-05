from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    FAILED = "failed"
    COMPLETED = "completed"


class Stage(StrEnum):
    PREPROCESS = "preprocess"
    INTERPRET = "interpret"
    POSTPROCESS = "postprocess"
    QA = "qa"
    REPORT = "report"


STAGES = list(Stage)


class ApprovalAction(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class FailureInjection(StrEnum):
    TIMEOUT = "timeout"
    FILE_MISSING = "file_missing"
    QA_FAILURE = "qa_failure"
    INVALID_PLANNER_OUTPUT = "invalid_planner_output"


class Artifact(BaseModel):
    uri: str
    sha256: str
    kind: str
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolContract(BaseModel):
    name: str
    version: str
    supported_sensor_types: list[str]
    input_kinds: list[str]
    output_kinds: list[str]
    applicable_stages: list[Stage] = Field(default_factory=list)
    supported_task_types: list[str] = Field(default_factory=lambda: ["water_extraction"])
    parameter_names: list[str] = Field(default_factory=list)
    retry_limit: int = Field(default=2, ge=0)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    fallback_tool: str | None = None
    rationale: str | None = None


class QualityResult(BaseModel):
    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ApprovalEvent(BaseModel):
    action: ApprovalAction
    actor: str
    comment: str | None = None
    stage: Stage | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Attempt(BaseModel):
    step_id: str
    stage: Stage
    tool: str
    idempotency_key: str
    status: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    error: str | None = None


class JobRequest(BaseModel):
    task_type: Literal["water_extraction", "building_extraction"] = "water_extraction"
    sensor_type: Literal["optical", "sar"] = "optical"
    image_uri: str = "mock://input"
    # Zero-based band indices. Optical defaults: green=1, nir=3 for four-band imagery.
    band_indices: dict[str, int] = Field(default_factory=dict)
    # Test/demo only: causes one recoverable failure at this stage.
    fail_once_stage: Stage | None = None
    failure_injections: list[FailureInjection] = Field(default_factory=list)
    injection_stage: Stage = Stage.INTERPRET


class Job(BaseModel):
    job_id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.PENDING
    stage: Stage = Stage.PREPROCESS
    request: JobRequest
    image_manifest: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    plans: dict[str, ExecutionPlan] = Field(default_factory=dict)
    attempts: list[Attempt] = Field(default_factory=list)
    quality: QualityResult | None = None
    approval: ApprovalEvent | None = None
    memory_refs: list[str] = Field(default_factory=list)
    completed_stages: list[Stage] = Field(default_factory=list)
    injected_failures: list[Stage] = Field(default_factory=list)
    injected_failure_keys: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
