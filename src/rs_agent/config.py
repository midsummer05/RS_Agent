from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    artifact_dir: Path
    log_level: str
    deepseek_api_key: str | None
    deepseek_model: str | None
    deepseek_base_url: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            state_dir=Path(os.getenv("RS_AGENT_STATE_DIR", "state")),
            artifact_dir=Path(os.getenv("RS_AGENT_ARTIFACT_DIR", "artifacts")),
            log_level=os.getenv("RS_AGENT_LOG_LEVEL", "INFO"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_model=os.getenv("DEEPSEEK_MODEL") or None,
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
