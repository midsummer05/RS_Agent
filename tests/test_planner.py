from rs_agent.domain import Artifact, Job, JobRequest, Stage
from rs_agent.planning import Planner


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return next(self.responses)


def test_planner_repairs_an_invalid_profile_once_then_uses_a_registered_profile():
    client = ScriptedClient(
            [
            '{"tool":"sar_adaptive_threshold","parameters":{"percentile":35}}',
            '{"tool":"optical_ndwi","parameters":{"threshold":0.1},"rationale":"low contrast water"}',
        ]
    )
    planner = Planner(client=client)
    job = Job(request=JobRequest(sensor_type="optical"))
    plan = planner.plan(Stage.INTERPRET, job)
    assert plan.tool == "optical_ndwi"
    assert plan.parameters["threshold"] == 0.1
    assert plan.fallback_tool == "optical_ndwi"
    assert len(client.calls) == 2


def test_planner_calls_llm_at_all_four_documented_planning_points():
    client = ScriptedClient(
        [
            '{"tool":"image_adapter_normalize","parameters":{}}',
            '{"tool":"sar_adaptive_threshold","parameters":{"percentile":35}}',
            '{"tool":"morphology_and_polygonize","parameters":{"min_component_pixels":9}}',
                '{"tool":"mask_statistics_and_geometry_qa","parameters":{"min_coverage_fraction":0,"max_coverage_fraction":0.98}}',
        ]
    )
    planner = Planner(client=client)
    job = Job(request=JobRequest(sensor_type="sar"))
    for stage in (Stage.PREPROCESS, Stage.INTERPRET, Stage.POSTPROCESS, Stage.QA):
        assert planner.plan(stage, job).fallback_tool is not None
    assert len(client.calls) == 4


def test_planner_context_has_artifact_references_and_diagnostics_not_raw_data():
    client = ScriptedClient(['{"tool":"image_adapter_normalize","parameters":{}}'])
    planner = Planner(client=client)
    job = Job(request=JobRequest(), image_manifest={"width": 1024, "height": 512})
    job.artifacts.append(Artifact(uri="artifact://abc", sha256="abc", kind="preprocessed_array"))
    context = planner.context(Stage.INTERPRET, job)
    assert context["artifacts"][0]["uri"] == "artifact://abc"
    assert context["routing_policy"]["decision_type"] == "tool_and_registered_parameter_profile"
    assert context["routing_policy"]["registered_parameter_profiles"]
    assert "pixel_values" not in str(context)


def test_planner_accepts_registered_optical_building_profile():
    planner = Planner(
        client=ScriptedClient(
            ['{"tool":"optical_built_index","parameters":{"threshold":0.0},"rationale":"SWIR and NIR diagnostics support balanced built-index"}']
        )
    )
    plan = planner.plan(
        Stage.INTERPRET,
        Job(request=JobRequest(task_type="building_extraction", sensor_type="optical")),
    )
    assert plan.tool == "optical_built_index"
    assert plan.parameters == {"threshold": 0.0}
    assert plan.fallback_tool == "optical_built_index"
