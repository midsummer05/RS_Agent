from __future__ import annotations

from typing import Any

from rs_agent.domain import ExecutionPlan, Stage, ToolContract


class ToolRegistry:
    """Machine-readable allowed tool surface; plans are rejected before execution."""

    def __init__(self, contracts: list[ToolContract] | None = None) -> None:
        self._contracts = {contract.name: contract for contract in contracts or self._defaults()}

    @staticmethod
    def _defaults() -> list[ToolContract]:
        common = {
            "supported_sensor_types": ["optical", "sar"],
            "supported_task_types": ["water_extraction", "building_extraction"],
        }
        return [
            ToolContract(
                name="image_adapter_normalize",
                version="0.1.0",
                input_kinds=["image"],
                output_kinds=["preprocessed_array"],
                applicable_stages=[Stage.PREPROCESS],
                parameter_names=[],
                **common,
            ),
            ToolContract(
                name="optical_built_index",
                version="0.1.0",
                input_kinds=["preprocessed_array"],
                output_kinds=["raw_mask"],
                applicable_stages=[Stage.INTERPRET],
                supported_sensor_types=["optical"],
                supported_task_types=["building_extraction"],
                parameter_names=["threshold"],
            ),
            ToolContract(
                name="optical_ndwi",
                version="0.1.0",
                input_kinds=["preprocessed_array"],
                output_kinds=["raw_mask"],
                applicable_stages=[Stage.INTERPRET],
                supported_sensor_types=["optical"],
                supported_task_types=["water_extraction"],
                parameter_names=["threshold"],
            ),
            ToolContract(
                name="sar_adaptive_threshold",
                version="0.1.0",
                input_kinds=["preprocessed_array"],
                output_kinds=["raw_mask"],
                applicable_stages=[Stage.INTERPRET],
                supported_sensor_types=["sar"],
                supported_task_types=["water_extraction"],
                parameter_names=["percentile"],
            ),
            ToolContract(
                name="morphology_and_polygonize",
                version="0.1.0",
                input_kinds=["raw_mask"],
                output_kinds=["classification_mask_geotiff", "classification_vectors_geojson"],
                applicable_stages=[Stage.POSTPROCESS],
                parameter_names=["min_component_pixels"],
                **common,
            ),
            ToolContract(
                name="mask_statistics_and_geometry_qa",
                version="0.1.0",
                input_kinds=["classification_mask_geotiff"],
                output_kinds=["statistics_json"],
                applicable_stages=[Stage.QA],
                parameter_names=["min_coverage_fraction", "max_coverage_fraction"],
                **common,
            ),
            ToolContract(
                name="markdown_report",
                version="0.1.0",
                input_kinds=["statistics_json"],
                output_kinds=["report_markdown"],
                applicable_stages=[Stage.REPORT],
                parameter_names=[],
                **common,
            ),
        ]

    def contracts_for(self, stage: Stage, sensor_type: str, task_type: str) -> list[ToolContract]:
        return [
            contract
            for contract in self._contracts.values()
            if stage in contract.applicable_stages
            and sensor_type in contract.supported_sensor_types
            and task_type in contract.supported_task_types
        ]

    def context_contracts(
        self, stage: Stage, sensor_type: str, task_type: str
    ) -> list[dict[str, Any]]:
        return [
            contract.model_dump(mode="json")
            for contract in self.contracts_for(stage, sensor_type, task_type)
        ]

    @staticmethod
    def _profiles() -> dict[str, list[dict[str, Any]]]:
        """Finite, reviewed parameter profiles available to the Planner.

        A profile is an execution-safe parameter configuration, not a per-image
        prompt-time numeric value. This keeps Phase 2 parameter planning
        auditable while still allowing the LLM to choose a stage strategy.
        """
        return {
            "image_adapter_normalize": [
                {"name": "standard", "parameters": {}, "when": "default input normalization"}
            ],
            "optical_ndwi": [
                {"name": "balanced", "parameters": {"threshold": 0.0}, "when": "default"},
                {"name": "conservative_water", "parameters": {"threshold": 0.1}, "when": "avoid weak water candidates"},
                {"name": "permissive_water", "parameters": {"threshold": -0.1}, "when": "retain low-contrast water candidates"},
            ],
            "optical_built_index": [
                {"name": "balanced", "parameters": {"threshold": 0.0}, "when": "default built-index separation"},
                {"name": "conservative_buildings", "parameters": {"threshold": 0.1}, "when": "avoid bright non-building surfaces"},
                {"name": "permissive_buildings", "parameters": {"threshold": -0.1}, "when": "retain weak built-up candidates"},
            ],
            "sar_adaptive_threshold": [
                {"name": "conservative_water", "parameters": {"percentile": 25.0}, "when": "avoid broad low-backscatter masks"},
                {"name": "balanced", "parameters": {"percentile": 35.0}, "when": "default"},
                {"name": "permissive_water", "parameters": {"percentile": 45.0}, "when": "retain fragmented water candidates"},
            ],
            "morphology_and_polygonize": [
                {"name": "detail_preserving", "parameters": {"min_component_pixels": 4.0}, "when": "small valid objects expected"},
                {"name": "balanced", "parameters": {"min_component_pixels": 9.0}, "when": "default"},
                {"name": "noise_reduction", "parameters": {"min_component_pixels": 25.0}, "when": "speckle/noise dominates"},
            ],
            "mask_statistics_and_geometry_qa": [
                {"name": "strict", "parameters": {"min_coverage_fraction": 0.0, "max_coverage_fraction": 0.9}, "when": "dominant masks are implausible"},
                {"name": "balanced", "parameters": {"min_coverage_fraction": 0.0, "max_coverage_fraction": 0.98}, "when": "default"},
                {"name": "permissive", "parameters": {"min_coverage_fraction": 0.0, "max_coverage_fraction": 0.995}, "when": "large water extent is expected"},
            ],
        }

    def profiles_for(self, stage: Stage, sensor_type: str, task_type: str) -> list[dict[str, Any]]:
        return [
            profile
            for contract in self.contracts_for(stage, sensor_type, task_type)
            for profile in self._profiles().get(contract.name, [])
        ]

    def validate_profile(
        self, stage: Stage, sensor_type: str, task_type: str, plan: ExecutionPlan
    ) -> None:
        self.validate(stage, sensor_type, task_type, plan)
        allowed = self._profiles().get(plan.tool, [])
        if not any(plan.parameters == profile["parameters"] for profile in allowed):
            raise ValueError(f"Parameters for {plan.tool} must match a registered profile")

    def validate(self, stage: Stage, sensor_type: str, task_type: str, plan: ExecutionPlan) -> None:
        contract = self._contracts.get(plan.tool)
        if not contract:
            raise ValueError(f"Unknown tool: {plan.tool}")
        if stage not in contract.applicable_stages:
            raise ValueError(f"Tool {plan.tool} is not allowed in {stage}")
        if (
            sensor_type not in contract.supported_sensor_types
            or task_type not in contract.supported_task_types
        ):
            raise ValueError(f"Tool {plan.tool} is incompatible with request")
        unexpected = set(plan.parameters) - set(contract.parameter_names)
        if unexpected:
            raise ValueError(f"Unsupported parameters for {plan.tool}: {sorted(unexpected)}")
        for name, value in plan.parameters.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {name} must be numeric")
            if name == "threshold" and not -1 <= value <= 1:
                raise ValueError("threshold must be between -1 and 1")
            if name == "percentile" and not 1 <= value <= 99:
                raise ValueError("percentile must be between 1 and 99")
            if name == "min_component_pixels" and value < 1:
                raise ValueError("min_component_pixels must be positive")
            if name.endswith("coverage_fraction") and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if plan.fallback_tool:
            fallback = self._contracts.get(plan.fallback_tool)
            if (
                not fallback
                or stage not in fallback.applicable_stages
                or sensor_type not in fallback.supported_sensor_types
                or task_type not in fallback.supported_task_types
            ):
                raise ValueError("fallback_tool is not compatible with this stage")
