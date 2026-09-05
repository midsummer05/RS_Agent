FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV RS_AGENT_STATE_DIR=/data/state
ENV RS_AGENT_ARTIFACT_DIR=/data/artifacts
CMD ["uvicorn", "rs_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
