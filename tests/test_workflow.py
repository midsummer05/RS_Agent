from rs_agent.domain import Job, JobRequest, JobStatus, Stage
from rs_agent.orchestration import WorkflowEngine
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore


def make_engine(tmp_path):
    return WorkflowEngine(
        SQLiteStateStore(tmp_path / "state" / "jobs.sqlite3"),
        LocalArtifactStore(tmp_path / "artifacts"),
    )


def test_mock_workflow_completes_and_persists_artifacts(tmp_path):
    engine = make_engine(tmp_path)
    job = engine.submit(Job(request=JobRequest()))
    done = engine.run(job.job_id)
    assert done.status is JobStatus.COMPLETED
    assert done.completed_stages == list(Stage)
    assert len(done.artifacts) == 5
    assert done.quality and done.quality.passed


def test_failure_then_resume_does_not_repeat_completed_tool(tmp_path):
    engine = make_engine(tmp_path)
    job = engine.submit(Job(request=JobRequest(fail_once_stage=Stage.INTERPRET)))
    failed = engine.run(job.job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.completed_stages == [Stage.PREPROCESS]
    # A fresh engine models worker process/container restart against persisted state.
    resumed = make_engine(tmp_path).resume(job.job_id)
    assert resumed.status is JobStatus.COMPLETED
    completed = [a for a in resumed.attempts if a.status == "completed"]
    assert [a.stage for a in completed].count(Stage.PREPROCESS) == 1
    assert [a.stage for a in completed].count(Stage.INTERPRET) == 1


def test_checkpoints_exist_for_failure_and_recovery(tmp_path):
    engine = make_engine(tmp_path)
    job = engine.submit(Job(request=JobRequest(fail_once_stage=Stage.POSTPROCESS)))
    engine.run(job.job_id)
    engine.resume(job.job_id)
    reasons = [item["reason"] for item in engine.state.checkpoints(job.job_id)]
    assert "recoverable_error" in reasons
    assert reasons[-1] == "completed"


def test_stage_transition_validation_rejects_out_of_order_stage(tmp_path):
    engine = make_engine(tmp_path)
    job = Job(request=JobRequest())
    try:
        engine._validate_next_stage(job, Stage.QA)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-order stage must be rejected")
