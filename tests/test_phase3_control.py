from rs_agent.domain import ApprovalAction, Job, JobRequest, JobStatus, Stage
from rs_agent.orchestration import WorkflowEngine
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore


def make_engine(tmp_path):
    return WorkflowEngine(
        SQLiteStateStore(tmp_path / "state" / "jobs.sqlite3"),
        LocalArtifactStore(tmp_path / "artifacts"),
    )


def test_qa_failure_waits_for_human_and_approval_resumes(tmp_path):
    engine = make_engine(tmp_path)
    request = JobRequest(failure_injections=["qa_failure"], injection_stage=Stage.QA)
    job = engine.submit(Job(request=request))
    waiting = engine.run(job.job_id)
    assert waiting.status is JobStatus.WAITING_HUMAN
    approved = engine.apply_approval(
        job.job_id, ApprovalAction.APPROVE, "reviewer", "acceptable for demo"
    )
    assert approved.status is JobStatus.PENDING
    finished = engine.resume(job.job_id)
    assert finished.status is JobStatus.COMPLETED
    event_names = [item["event_type"] for item in engine.state.traces(job.job_id)]
    assert "human_approval" in event_names
    assert "tool_completed" in event_names


def test_edit_restarts_from_requested_stage_with_validated_parameters(tmp_path):
    engine = make_engine(tmp_path)
    job = Job(
        request=JobRequest(image_uri="/not-read-yet.tif"),
        status=JobStatus.WAITING_HUMAN,
        stage=Stage.QA,
        completed_stages=[Stage.PREPROCESS, Stage.INTERPRET, Stage.POSTPROCESS, Stage.QA],
    )
    engine.submit(job)
    edited = engine.apply_approval(
        job.job_id,
        ApprovalAction.EDIT,
        "reviewer",
        stage=Stage.INTERPRET,
        parameters={"threshold": 0.2},
    )
    assert edited.status is JobStatus.PENDING
    assert edited.completed_stages == [Stage.PREPROCESS]
    assert edited.plans["interpret"].parameters["threshold"] == 0.2


def test_timeout_injection_is_checkpointed_and_can_resume(tmp_path):
    engine = make_engine(tmp_path)
    job = engine.submit(
        Job(request=JobRequest(failure_injections=["timeout"], injection_stage=Stage.INTERPRET))
    )
    assert engine.run(job.job_id).status is JobStatus.FAILED
    assert engine.resume(job.job_id).status is JobStatus.COMPLETED
    reasons = [checkpoint["reason"] for checkpoint in engine.state.checkpoints(job.job_id)]
    assert "recoverable_timeout" in reasons
