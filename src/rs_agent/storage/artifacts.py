from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rs_agent.domain import Artifact


class LocalArtifactStore:
    """Content-addressed, append-only artifacts; state keeps only URI/hash/metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, kind: str, payload: dict[str, Any]) -> Artifact:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return self.put_bytes(kind, raw, "json")

    def put_bytes(self, kind: str, raw: bytes, extension: str) -> Artifact:
        digest = hashlib.sha256(raw).hexdigest()
        target = self.root / digest[:2] / f"{digest}.{extension.lstrip('.')}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        return Artifact(
            uri=f"artifact://{digest}",
            sha256=digest,
            kind=kind,
            metadata={"path": str(target), "extension": extension.lstrip(".")},
        )

    @staticmethod
    def path(artifact: Artifact) -> Path:
        return Path(str(artifact.metadata["path"]))
