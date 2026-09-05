from __future__ import annotations

from typing import Any

from rs_agent.domain import ExecutionPlan, Stage, ToolContract


class MockToolRegistry:
    """Deterministic tools used to establish workflow semantics before image I/O exists."""

    def __init__(self) -> None:
        self.contracts = {
            stage: ToolContract(
                name=f"mock_{stage.value}",
                version="0.1.0",
                supported_sensor_types=["optical", "sar"],
                input_kinds=["mock"],
                output_kinds=[f"{stage.value}_result"],
            )
            for stage in Stage
        }

    def plan(self, stage: Stage) -> ExecutionPlan:
        return ExecutionPlan(
            tool=self.contracts[stage].name,
            parameters={"deterministic": True},
            rationale="Phase 0 mock",
        )

    def run(self, stage: Stage, job_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job_id,
            "stage": stage.value,
            "tool": self.contracts[stage].name,
        }
        if stage is Stage.INTERPRET:
            payload["mask"] = "mock-mask-001"
        elif stage is Stage.QA:
            payload["quality_score"] = 0.98
        elif stage is Stage.REPORT:
            payload["summary"] = "Mock workflow completed successfully."
        else:
            payload["result"] = f"mock-{stage.value}-complete"
        return payload
