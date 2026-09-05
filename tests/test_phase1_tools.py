import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from rs_agent.domain import Job, JobRequest, JobStatus
from rs_agent.orchestration import WorkflowEngine
from rs_agent.storage import LocalArtifactStore, SQLiteStateStore


def test_optical_geotiff_produces_mask_vector_stats_and_report(tmp_path):
    source = tmp_path / "optical.tif"
    # Four bands: RGB + NIR. The top half has green >> NIR and should be water.
    raster = np.full((4, 20, 20), 100, dtype="uint16")
    raster[1, :10, :] = 900
    raster[3, :10, :] = 50
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=4,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 200, 10, 10),
    ) as dst:
        dst.write(raster)
    engine = WorkflowEngine(
        SQLiteStateStore(tmp_path / "state" / "jobs.sqlite3"),
        LocalArtifactStore(tmp_path / "artifacts"),
    )
    done = engine.run(
        engine.submit(Job(request=JobRequest(image_uri=str(source), sensor_type="optical"))).job_id
    )
    assert done.status is JobStatus.COMPLETED
    kinds = {artifact.kind for artifact in done.artifacts}
    assert {
        "water_mask_geotiff",
        "water_vectors_geojson",
        "statistics_json",
        "report_markdown",
    } <= kinds
    stats = next(a for a in done.artifacts if a.kind == "statistics_json")
    assert json.loads(LocalArtifactStore.path(stats).read_text())["water_pixels"] > 0
    assert "ndwi_percentiles" in done.image_manifest["routing_diagnostics"]


def test_sar_geotiff_uses_low_backscatter_water_route(tmp_path):
    source = tmp_path / "sar.tif"
    raster = np.full((1, 20, 20), 0.9, dtype="float32")
    raster[:, :8, :] = 0.05
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 200, 10, 10),
    ) as dst:
        dst.write(raster)
    engine = WorkflowEngine(
        SQLiteStateStore(tmp_path / "state" / "jobs.sqlite3"),
        LocalArtifactStore(tmp_path / "artifacts"),
    )
    done = engine.run(
        engine.submit(Job(request=JobRequest(image_uri=str(source), sensor_type="sar"))).job_id
    )
    assert done.status is JobStatus.COMPLETED
    assert done.quality is not None
    assert "vv_percentiles" in done.image_manifest["routing_diagnostics"]


def test_optical_multispectral_geotiff_produces_building_artifacts(tmp_path):
    source = tmp_path / "buildings.tif"
    raster = np.full((13, 20, 20), 100, dtype="uint16")
    raster[11, :10, :] = 900  # SWIR high relative to NIR: built-index positive.
    with rasterio.open(source, "w", driver="GTiff", height=20, width=20, count=13,
                       dtype="uint16", crs="EPSG:3857", transform=from_origin(0, 200, 10, 10)) as dst:
        dst.write(raster)
    engine = WorkflowEngine(SQLiteStateStore(tmp_path / "state" / "jobs.sqlite3"), LocalArtifactStore(tmp_path / "artifacts"))
    done = engine.run(engine.submit(Job(request=JobRequest(task_type="building_extraction", sensor_type="optical", image_uri=str(source), band_indices={"nir": 7, "swir": 11}))).job_id)
    assert done.status is JobStatus.COMPLETED
    assert "building_mask_geotiff" in {artifact.kind for artifact in done.artifacts}
