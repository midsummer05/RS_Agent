from rs_agent.domain import Artifact, Job, JobRequest, Stage, ToolContract
from rs_agent.planning import Planner, ToolRegistry


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return next(self.responses)


def test_planner_uses_llm_only_when_the_registry_has_a_real_tool_choice():
    client = ScriptedClient(
            [
                '{"tool":"sar_adaptive_threshold","parameters":{"percentile":35}}',
                '{"tool":"optical_ndwi_conservative","parameters":{"threshold":0.12},"rationale":"metadata supports conservative optical route"}',
        ]
    )
    contracts = ToolRegistry._defaults()
    contracts.append(
        ToolContract(
            name="optical_ndwi_conservative",
            version="0.1.0",
            supported_sensor_types=["optical"],
            supported_task_types=["water_extraction"],
            input_kinds=["preprocessed_array"],
            output_kinds=["raw_mask"],
            applicable_stages=[Stage.INTERPRET],
            parameter_names=["threshold"],
        )
    )
    planner = Planner(registry=ToolRegistry(contracts), client=client)
    job = Job(request=JobRequest(sensor_type="optical"))
    plan = planner.plan(Stage.INTERPRET, job)
    assert plan.tool == "optical_ndwi_conservative"
    assert plan.parameters["threshold"] == 0.12
    assert len(client.calls) == 2


def test_single_eligible_tool_defers_to_fixed_rule_profile_without_an_llm_call():
    client = ScriptedClient(['{"tool":"not_registered","parameters":{}}'] * 2)
    planner = Planner(client=client)
    plan = planner.plan(Stage.INTERPRET, Job(request=JobRequest(sensor_type="sar")))
    assert plan.tool == "sar_adaptive_threshold"
    assert plan.parameters == {"percentile": 35.0}
    assert client.calls == []
    assert planner.last_decision["source"] == "rule_single_eligible_route"


def test_planner_context_has_artifact_references_and_diagnostics_not_raw_data():
    client = ScriptedClient(['{"tool":"image_adapter_normalize","parameters":{}}'])
    planner = Planner(client=client)
    job = Job(request=JobRequest(), image_manifest={"width": 1024, "height": 512})
    job.artifacts.append(Artifact(uri="artifact://abc", sha256="abc", kind="preprocessed_array"))
    context = planner.context(Stage.INTERPRET, job)
    assert context["artifacts"][0]["uri"] == "artifact://abc"
    assert context["routing_policy"]["decision_type"] == "tool_choice_only"
    assert "pixel_values" not in str(context)
