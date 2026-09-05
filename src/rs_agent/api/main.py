from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rs_agent.config import Settings, configure_logging
from rs_agent.domain import ApprovalAction, Job, JobRequest, Stage
from rs_agent.orchestration import WorkflowEngine
from rs_agent.planning import DeepSeekClient, Planner
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore

app = FastAPI(title="RS Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)
settings = Settings.from_env()
configure_logging(settings.log_level)


@lru_cache
def engine() -> WorkflowEngine:
    client = None
    if settings.deepseek_api_key and settings.deepseek_model:
        client = DeepSeekClient(
            settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url
        )
    return WorkflowEngine(
        SQLiteStateStore(settings.state_dir / "jobs.sqlite3"),
        LocalArtifactStore(settings.artifact_dir),
        planner=Planner(client=client),
    )


class JobCreated(BaseModel):
    job_id: UUID


class ApprovalInput(BaseModel):
    action: ApprovalAction
    actor: str = "api-user"
    comment: str | None = None
    stage: Stage | None = None
    parameters: dict[str, float] = Field(default_factory=dict)


class UploadCreated(BaseModel):
    image_uri: str
    filename: str
    size_bytes: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreated, status_code=201)
def create_job(request: JobRequest) -> JobCreated:
    job = engine().submit(Job(request=request))
    return JobCreated(job_id=job.job_id)


@app.post("/uploads", response_model=UploadCreated, status_code=201)
async def upload_image(file: Annotated[UploadFile, File(...)]) -> UploadCreated:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(415, "supported image types: GeoTIFF, PNG, JPEG")
    upload_dir = engine().artifacts.root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}{suffix}"
    size = 0
    with destination.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 200 * 1024 * 1024:
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "image exceeds 200 MB upload limit")
            target.write(chunk)
    return UploadCreated(image_uri=str(destination), filename=file.filename or destination.name, size_bytes=size)


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID) -> Job:
    job = engine().state.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}/checkpoints")
def get_checkpoints(job_id: UUID) -> list[dict[str, str]]:
    return engine().state.checkpoints(job_id)


@app.get("/jobs/{job_id}/traces")
def get_traces(job_id: UUID) -> list[dict[str, object]]:
    return engine().state.traces(job_id)


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: UUID) -> StreamingResponse:
    if engine().state.get(job_id) is None:
        raise HTTPException(404, "job not found")

    async def stream():
        previous = ""
        while True:
            job = engine().state.get(job_id)
            if job is None:
                return
            current = job.model_dump_json()
            if current != previous:
                previous = current
                yield f"event: job\ndata: {current}\n\n"
            if job.status.value in {"completed", "failed", "waiting_human"}:
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/artifacts/{sha256}")
def download_artifact(job_id: UUID, sha256: str) -> FileResponse:
    job = engine().state.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    artifact = next((item for item in job.artifacts if item.sha256 == sha256), None)
    if artifact is None:
        raise HTTPException(404, "artifact not found")
    path = engine().artifacts.path(artifact)
    if not path.is_file():
        raise HTTPException(404, "artifact file is unavailable")
    return FileResponse(path, filename=f"{artifact.kind}.{artifact.metadata.get('extension', 'bin')}")


@app.get("/memory/{sensor_type}/{task_type}")
def get_experience_memory(sensor_type: str, task_type: str) -> list[dict[str, object]]:
    return engine().state.memories(sensor_type, task_type)


@app.post("/jobs/{job_id}/resume", response_model=Job)
def resume_job(job_id: UUID) -> Job:
    try:
        return engine().resume(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/jobs/{job_id}/approval", response_model=Job)
def approve_job(job_id: UUID, request: ApprovalInput) -> Job:
    try:
        return engine().apply_approval(
            job_id,
            request.action,
            request.actor,
            request.comment,
            request.stage,
            request.parameters,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
