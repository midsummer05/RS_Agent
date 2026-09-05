"""Fetch a fixed research-evaluation subset from the publisher's public bucket.

No training, label-based selection, or per-case parameter tuning. STAC explicitly
declares proprietary: do not describe this corpus as CC/open-licensed or redistribute.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from rs_agent.tools.image_adapter import ImageAdapter  # noqa: F401 -- registers wheel PROJ data

ROOT = Path(__file__).resolve().parents[1] / "data"
BASE = "https://storage.googleapis.com/sen1floods11/v1.1/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(relative):
    destination = ROOT / "sources" / "sen1floods11-v1.1" / relative
    if not destination.exists():
        with httpx.Client(trust_env=False, timeout=120, follow_redirects=True) as client:
            response = client.get(BASE + relative)
            response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
    return destination


def select(split, count):
    path = fetch(f"splits/flood_handlabeled/flood_{split}_data.csv")
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            chip = line.split(",")[0].removesuffix("_S1Hand.tif")
            groups[chip.split("_")[0]].append(chip)
    ordered = [
        chip
        for i in range(max(map(len, groups.values())))
        for event in sorted(groups)
        for chip in sorted(groups[event])[i : i + 1]
    ]
    return ordered[:count]


def prepare(item, metadata):
    chip, split, resolution = item
    paths = {
        layer: fetch(f"data/flood_events/HandLabeled/{layer}/{chip}_{layer}.tif")
        for layer in ("S1Hand", "S2Hand", "LabelHand")
    }
    with rasterio.open(paths["LabelHand"]) as src:
        lon, lat = (
            (src.bounds.left + src.bounds.right) / 2,
            (src.bounds.bottom + src.bounds.top) / 2,
        )
        crs = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds, resolution=resolution
        )
    derived = ROOT / "samples" / "sen1floods11-v1.1" / chip
    derived.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for layer, path in paths.items():
        output = derived / f"{layer}_{resolution}m.tif"
        with rasterio.open(path) as src:
            is_label = layer == "LabelHand"
            nodata = -1 if is_label else -9999
            pixels = np.full(
                (src.count, height, width), nodata, dtype="int16" if is_label else "float32"
            )
            for band in range(src.count):
                reproject(
                    rasterio.band(src, band + 1),
                    pixels[band],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=nodata,
                    resampling=Resampling.nearest if is_label else Resampling.bilinear,
                )
            with rasterio.open(
                output,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=src.count,
                dtype=pixels.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress="deflate",
            ) as dst:
                dst.write(pixels)
        outputs[layer] = output
    for sensor, layer in (("optical", "S2Hand"), ("sar", "S1Hand")):
        case_id = f"{split}_{chip}_{sensor}_{resolution}m"
        with rasterio.open(outputs[layer]) as src:
            bounds = list(src.bounds)
        event = chip.split("_")[0]
        info = metadata["Cambodia" if event == "Mekong" else event]
        manifest = {
            "case_id": case_id,
            "split": split,
            "sensor_type": sensor,
            "image_uri": "../" + outputs[layer].relative_to(ROOT).as_posix(),
            "label_uri": "../" + outputs["LabelHand"].relative_to(ROOT).as_posix(),
            "source": {
                "name": "Sen1Floods11 hand-labelled v1.1",
                "url": BASE,
                "license": "proprietary (publisher STAC); public research access; redistribution rights not established",
            },
            "acquisition_date": info["s2_date" if sensor == "optical" else "s1_date"].replace(
                "/", "-"
            ),
            "bands": {"green": 2, "nir": 7, "red": 3, "blue": 1}
            if sensor == "optical"
            else {"vv": 0, "vh": 1},
            "crs": crs,
            "resolution": resolution,
            "bounds": bounds,
            "image_sha256": digest(outputs[layer]),
            "label_sha256": digest(outputs["LabelHand"]),
            "expected_interpret_tool": "optical_ndwi"
            if sensor == "optical"
            else "sar_adaptive_threshold",
            "dataset_version": "phase4-sen1floods11-v1.1-subset-v1",
            "baseline_metrics": {
                "reference": f"../baselines/{case_id}.json",
                "algorithm_version": "deterministic-water-v2",
            },
            "label_classes": {"-1": "ignore", "0": "non-water", "1": "water"},
            "provenance": {
                "chip": chip,
                "original_split": "test" if split == "evaluation" else "valid",
                "crop": "full original chip; no spatial cropping",
                "native_ground_resolution_m": 10,
                "resampling": "UTM bilinear imagery / nearest labels; 10/20/30m are derived, not native sensor grades",
                "source_files": {
                    k: {
                        "url": BASE
                        + v.relative_to(ROOT / "sources" / "sen1floods11-v1.1").as_posix(),
                        "sha256": digest(v),
                    }
                    for k, v in paths.items()
                },
            },
        }
        (ROOT / "manifests" / f"{case_id}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"prepared {chip}: {split}, {resolution}m", flush=True)


def main():
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    for kind in ("source", "label"):
        path = fetch(f"catalog/sen1floods11_hand_labeled_{kind}/collection.json")
        assert json.loads(path.read_text())["license"] == "proprietary"
    metadata = {
        f["properties"]["location"]: f["properties"]
        for f in json.loads(fetch("Sen1Floods11_Metadata.geojson").read_text())["features"]
    }
    test, smoke = select("test", 12), select("valid", 2)
    assert not set(test) & set(smoke)
    selection = [(c, "evaluation", (10, 20, 30)[i % 3]) for i, c in enumerate(test)] + [
        (c, "smoke", 10) for c in smoke
    ]
    lock = ROOT / "phase4-selection.json"
    payload = {
        "version": 1,
        "selection_method": "round-robin sorted events, lexicographic chip IDs, before reading labels",
        "cases": selection,
    }
    if lock.exists() and json.loads(lock.read_text()) != json.loads(json.dumps(payload)):
        raise ValueError("Frozen selection differs; create a new version explicitly")
    lock.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: prepare(item, metadata), selection))


if __name__ == "__main__":
    main()
