from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DataSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    license: str


class EvaluationCase(BaseModel):
    """Versioned contract for a single immutable evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: Literal["evaluation", "smoke"] = "evaluation"
    task_type: Literal["water_extraction", "building_extraction"] = "water_extraction"
    sensor_type: Literal["optical", "sar"]
    image_uri: str
    label_uri: str
    source: DataSource
    acquisition_date: str | None = None
    bands: dict[str, int] = Field(default_factory=dict)
    crs: str | None = None
    resolution: float | None = Field(default=None, gt=0)
    bounds: list[float] | None = None
    image_sha256: str | None = None
    label_sha256: str | None = None
    expected_interpret_tool: str
    dataset_version: str = "unspecified"
    provenance: dict = Field(default_factory=dict)
    label_classes: dict[str, str] = Field(default_factory=lambda: {"0": "non-water", "1": "water"})
    baseline_metrics: dict = Field(default_factory=dict)

    def resolve_paths(self, manifest_path: Path) -> EvaluationCase:
        root = manifest_path.parent
        return self.model_copy(
            update={
                "image_uri": str((root / self.image_uri).resolve())
                if not Path(self.image_uri).is_absolute()
                else self.image_uri,
                "label_uri": str((root / self.label_uri).resolve())
                if not Path(self.label_uri).is_absolute()
                else self.label_uri,
            }
        )

    def validate_files(self) -> None:
        for label, value, expected_hash in (
            ("image", self.image_uri, self.image_sha256),
            ("label", self.label_uri, self.label_sha256),
        ):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"{label} file does not exist for {self.case_id}: {path}")
            if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"{label} checksum mismatch for {self.case_id}")
