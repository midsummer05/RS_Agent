from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

from rs_agent.domain.models import Job, JobStatus, utcnow


class SQLiteStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                    payload TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    stage TEXT NOT NULL, reason TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_memory (
                    sensor_type TEXT NOT NULL, task_type TEXT NOT NULL, memory_key TEXT NOT NULL,
                    payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(sensor_type, task_type, memory_key)
                );
            """)

    def save(self, job: Job, reason: str) -> None:
        job.updated_at = utcnow()
        payload = job.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs(job_id,status,stage,payload,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, stage=excluded.stage,
                payload=excluded.payload, updated_at=excluded.updated_at""",
                (str(job.job_id), job.status, job.stage, payload, job.updated_at.isoformat()),
            )
            conn.execute(
                "INSERT INTO checkpoints(job_id,stage,reason,payload,created_at) VALUES(?,?,?,?,?)",
                (str(job.job_id), job.stage, reason, payload, job.updated_at.isoformat()),
            )

    def get(self, job_id: UUID | str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id=?", (str(job_id),)).fetchone()
        return Job.model_validate_json(row["payload"]) if row else None

    def queued(self) -> Iterable[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM jobs WHERE status IN (?,?) ORDER BY updated_at",
                (JobStatus.PENDING, JobStatus.RUNNING),
            ).fetchall()
        return [Job.model_validate_json(row["payload"]) for row in rows]

    def checkpoints(self, job_id: UUID | str) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stage,reason,created_at FROM checkpoints WHERE job_id=? ORDER BY id",
                (str(job_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def trace(self, job_id: UUID | str, event_type: str, payload: dict[str, object]) -> None:
        """Trace payloads must already be summaries; never persist credentials or image arrays."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trace_events(job_id,event_type,payload,created_at) VALUES(?,?,?,?)",
                (
                    str(job_id),
                    event_type,
                    json.dumps(payload, sort_keys=True),
                    utcnow().isoformat(),
                ),
            )

    def traces(self, job_id: UUID | str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,event_type,payload,created_at FROM trace_events WHERE job_id=? ORDER BY id",
                (str(job_id),),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def remember(
        self, sensor_type: str, task_type: str, memory_key: str, payload: dict[str, object]
    ) -> str:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experience_memory(sensor_type,task_type,memory_key,payload,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(sensor_type,task_type,memory_key) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at""",
                (
                    sensor_type,
                    task_type,
                    memory_key,
                    json.dumps(payload, sort_keys=True),
                    utcnow().isoformat(),
                ),
            )
        return f"experience://{sensor_type}/{task_type}/{memory_key}"

    def memories(self, sensor_type: str, task_type: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_key,payload,updated_at FROM experience_memory WHERE sensor_type=? AND task_type=?",
                (sensor_type, task_type),
            ).fetchall()
        return [
            {
                "key": row["memory_key"],
                "payload": json.loads(row["payload"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
