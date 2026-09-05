"""Prepare a fixed 12-image, labelled building-evaluation subset.

The source PNGs have no supplied CRS.  They are converted to an explicitly
synthetic local metric GeoTIFF grid solely so the Agent can exercise its raster
and area contracts; the manifests do not claim a real geographic location.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from rs_agent.tools.image_adapter import ImageAdapter  # noqa: F401 -- registers wheel PROJ data

ROOT = Path(__file__).resolve().parents[1] / "data"
REPOSITORY = "giswqs/WHU-Building-Dataset"
REVISION = "main"
BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
DATASET_VERSION = "whu-building-hf-mirror-test12-v1"
CASE_IDS = [f"test_{number:04d}" for number in range(1, 13)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(relative: str) -> Path:
    destination = ROOT / "sources" / "whu-building-hf-mirror" / relative
    if not destination.exists():
        # Hugging Face may require an enterprise/local proxy, unlike the GCS
        # water source.  Credentials are neither read nor persisted here.
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.get(f"{BASE}/{relative}")
            response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
    return destination


def write_geotiff(source: Path, destination: Path, *, is_label: bool) -> None:
    array = np.asarray(Image.open(source))
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    else:
        array = np.moveaxis(array, -1, 0)
    if is_label:
        array = (array > 0).astype("uint8")
    height, width = array.shape[-2:]
    # 0.3 m is WHU's stated GSD.  The origin is synthetic because source PNGs
    # provide no location/CRS; it must never be used as a geographic assertion.
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": "EPSG:3857",
        "transform": from_origin(0, height * 0.3, 0.3, 0.3),
        "compress": "deflate",
        # Zero is the valid non-building class, never nodata.
        "nodata": None,
    }
    with rasterio.open(destination, "w", **profile) as output:
        output.write(array)


def main() -> None:
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    sample_root = ROOT / "samples" / DATASET_VERSION
    sample_root.mkdir(parents=True, exist_ok=True)
    selection = {
        "version": 1,
        "dataset_version": DATASET_VERSION,
        "selection_method": "first 12 lexicographic IDs in public test split; selection precedes download and label inspection",
        "case_ids": CASE_IDS,
    }
    selection_path = ROOT / "building-phase4-selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen building selection differs; create a new version explicitly")
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for item in CASE_IDS:
        manifest_path = ROOT / "manifests" / f"building_evaluation_whu_{item}.json"
        image = sample_root / f"{item}_image.tif"
        label = sample_root / f"{item}_label.tif"
        # A previous interrupted download must not make the frozen preparation
        # repeat already materialised cases.  The manifest hashes remain the
        # integrity check on every evaluation load.
        if manifest_path.exists() and image.is_file() and label.is_file():
            print(f"already prepared {item}", flush=True)
            continue
        image_source = fetch(f"test/Image/{item}.png")
        label_source = fetch(f"test/Mask/{item}.png")
        write_geotiff(image_source, image, is_label=False)
        write_geotiff(label_source, label, is_label=True)
        with rasterio.open(image) as dataset:
            manifest = {
                "case_id": f"building_evaluation_whu_{item}",
                "split": "evaluation",
                "task_type": "building_extraction",
                "sensor_type": "optical",
                "image_uri": "../" + image.relative_to(ROOT).as_posix(),
                "label_uri": "../" + label.relative_to(ROOT).as_posix(),
                "source": {
                    "name": "WHU Building Dataset (Hugging Face public mirror)",
                    "url": f"https://huggingface.co/datasets/{REPOSITORY}",
                    "license": "CC-BY-4.0",
                },
                "acquisition_date": None,
                "bands": {"red": 0, "green": 1, "blue": 2},
                "crs": str(dataset.crs),
                "resolution": 0.3,
                "bounds": list(dataset.bounds),
                "image_sha256": digest(image),
                "label_sha256": digest(label),
                "expected_interpret_tool": "optical_built_index",
                "dataset_version": DATASET_VERSION,
                "baseline_metrics": {
                    "reference": f"../baselines/building_evaluation_whu_{item}.json",
                    "algorithm_version": "deterministic-building-rgb-v1",
                },
                "label_classes": {"0": "non-building", "1": "building"},
                "provenance": {
                    "original_split": "test",
                    "source_tile": item,
                    "crop": "full 512x512 source tile; no crop or label-based selection",
                    "native_ground_resolution_m": 0.3,
                    "source_georeferencing": "not supplied with source PNG; local EPSG:3857 metric grid uses synthetic origin only for workflow compatibility, not a claimed location",
                    "source_files": {
                        "image": {"url": f"{BASE}/test/Image/{item}.png", "sha256": digest(image_source)},
                        "label": {"url": f"{BASE}/test/Mask/{item}.png", "sha256": digest(label_source)},
                    },
                },
            }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"prepared {item}", flush=True)


if __name__ == "__main__":
    main()
