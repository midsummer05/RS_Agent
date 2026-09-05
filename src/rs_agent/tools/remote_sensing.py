from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from rasterio.features import shapes
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label, median_filter

from rs_agent.domain import Artifact, ExecutionPlan, Job, QualityResult, Stage
from rs_agent.storage import LocalArtifactStore
from rs_agent.tools.image_adapter import ImageAdapter


@dataclass
class StageOutput:
    artifacts: list[Artifact]
    summary: dict[str, Any]
    quality: QualityResult | None = None


class DeterministicToolchain:
    """Phase 1 deterministic water-extraction route for optical and SAR imagery."""

    def __init__(self, adapter: ImageAdapter | None = None) -> None:
        self.adapter = adapter or ImageAdapter()

    def plan(self, stage: Stage, sensor_type: str) -> ExecutionPlan:
        tools = {
            Stage.PREPROCESS: "image_adapter_normalize",
            Stage.INTERPRET: "optical_ndwi"
            if sensor_type == "optical"
            else "sar_adaptive_threshold",
            Stage.POSTPROCESS: "morphology_and_polygonize",
            Stage.QA: "water_statistics_and_geometry_qa",
            Stage.REPORT: "markdown_report",
        }
        return ExecutionPlan(
            tool=tools[stage],
            parameters={"sensor_type": sensor_type},
            rationale="Phase 1 rule route",
        )

    @staticmethod
    def _artifact(job: Job, kind: str) -> Artifact:
        for artifact in reversed(job.artifacts):
            if artifact.kind == kind:
                return artifact
        raise ValueError(f"Required artifact missing: {kind}")

    def run(
        self, stage: Stage, job: Job, store: LocalArtifactStore, plan: ExecutionPlan | None = None
    ) -> StageOutput:
        parameters = plan.parameters if plan else {}
        if stage is Stage.PREPROCESS:
            image = self.adapter.read(job.request.image_uri)
            normalized = self.adapter.normalize(image.pixels)
            with TemporaryDirectory() as temp:
                npy = Path(temp) / "normalized.npy"
                np.save(npy, normalized)
                array_artifact = store.put_bytes("preprocessed_array", npy.read_bytes(), "npy")
            manifest = image.manifest()
            manifest["routing_diagnostics"] = self._routing_diagnostics(
                image.pixels, job.request.sensor_type, job.request.band_indices
            )
            job.image_manifest = manifest
            manifest_artifact = store.put_json("image_manifest", manifest)
            return StageOutput([array_artifact, manifest_artifact], manifest)
        if stage is Stage.INTERPRET:
            image = self.adapter.read(job.request.image_uri)
            normalized = np.load(store.path(self._artifact(job, "preprocessed_array")))
            mask, details = self._extract_water(
                image.pixels if image.bands >= 4 or job.request.sensor_type == "sar" else normalized,
                job.request.sensor_type, job.request.band_indices, parameters
            )
            nodata = image.source_profile.get("nodata")
            valid = np.isfinite(image.pixels).all(axis=-1)
            if nodata is not None:
                valid &= (image.pixels != nodata).all(axis=-1)
            mask &= valid
            return self._mask_output("raw_mask", mask, image, store, details)
        if stage is Stage.POSTPROCESS:
            image = self.adapter.read(job.request.image_uri)
            raw = self._read_mask(self._artifact(job, "raw_mask"))
            cleaned = self._cleanup(raw, int(parameters.get("min_component_pixels", 9)))
            mask_output = self._mask_output("water_mask_geotiff", cleaned, image, store, {})
            geojson = self._polygonize(cleaned, image)
            vector = store.put_json("water_vectors_geojson", geojson)
            return StageOutput(
                mask_output.artifacts + [vector], {"feature_count": len(geojson["features"])}
            )
        if stage is Stage.QA:
            image = self.adapter.read(job.request.image_uri)
            mask = self._read_mask(self._artifact(job, "water_mask_geotiff"))
            stats = self._statistics(mask, image)
            minimum = float(parameters.get("min_coverage_fraction", 0))
            maximum = float(parameters.get("max_coverage_fraction", 0.98))
            passed = minimum < stats["coverage_fraction"] < maximum
            quality = QualityResult(
                passed=passed,
                metrics={
                    "water_pixels": float(stats["water_pixels"]),
                    "coverage_fraction": stats["coverage_fraction"],
                    "water_area_m2": stats["water_area_m2"],
                },
                notes=[] if passed else ["Empty or implausibly dominant water mask"],
            )
            return StageOutput([store.put_json("statistics_json", stats)], stats, quality)
        stats = json.loads(store.path(self._artifact(job, "statistics_json")).read_text())
        report = self._report(job, stats)
        return StageOutput(
            [store.put_bytes("report_markdown", report.encode(), "md")], {"report": "generated"}
        )

    @staticmethod
    def _routing_diagnostics(
        pixels: np.ndarray, sensor: str, bands: dict[str, int]
    ) -> dict[str, Any]:
        """Small, trace-safe statistics available to a future multi-tool router."""
        valid = np.isfinite(pixels).all(axis=-1)
        result: dict[str, Any] = {"valid_pixel_fraction": float(valid.mean())}
        if not valid.any():
            return result
        if sensor == "optical":
            green = bands.get("green", 1 if pixels.shape[-1] >= 3 else 0)
            nir = bands.get("nir", 3 if pixels.shape[-1] >= 4 else -1)
            if 0 <= green < pixels.shape[-1] and 0 <= nir < pixels.shape[-1]:
                ndwi = (pixels[..., green] - pixels[..., nir]) / (
                    pixels[..., green] + pixels[..., nir] + 1e-6
                )
                values = ndwi[valid]
                result["ndwi_percentiles"] = {
                    "p02": float(np.percentile(values, 2)),
                    "p50": float(np.percentile(values, 50)),
                    "p98": float(np.percentile(values, 98)),
                }
                result["candidate_coverages"] = {
                    str(threshold): float((values > threshold).mean())
                    for threshold in (-0.1, 0.0, 0.1)
                }
        else:
            values = pixels[..., 0][valid]
            result["vv_percentiles"] = {
                "p02": float(np.percentile(values, 2)),
                "p50": float(np.percentile(values, 50)),
                "p98": float(np.percentile(values, 98)),
            }
            result["candidate_coverages"] = {
                f"p{percentile}": float((values <= np.percentile(values, percentile)).mean())
                for percentile in (25, 35, 45)
            }
        return result

    def _extract_water(
        self, image: np.ndarray, sensor: str, bands: dict[str, int], parameters: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, float]]:
        if sensor == "sar":
            denoised = median_filter(image[..., 0], size=3)
            percentile = float(parameters.get("percentile", 35))
            threshold = float(np.nanpercentile(denoised, percentile))
            return denoised <= threshold, {"threshold": threshold, "percentile": percentile}
        green_index = bands.get("green", 1 if image.shape[-1] >= 3 else 0)
        nir_index = bands.get("nir", 3 if image.shape[-1] >= 4 else -1)
        if (
            image.shape[-1] >= 4
            and 0 <= green_index < image.shape[-1]
            and 0 <= nir_index < image.shape[-1]
        ):
            green, nir = image[..., green_index], image[..., nir_index]
            index = (green - nir) / (green + nir + 1e-6)
            threshold = float(parameters.get("threshold", 0.0))
            return index > threshold, {"ndwi_threshold": threshold}
        # RGB fallback: water is relatively blue and dark; intended for demonstrations, not production labeling.
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        score = blue - (red + green) / 2
        threshold = float(np.nanpercentile(score, 65))
        return score >= threshold, {"blue_score_threshold": threshold}

    @staticmethod
    def _cleanup(mask: np.ndarray, min_component_pixels: int = 9) -> np.ndarray:
        cleaned = binary_opening(mask, structure=np.ones((3, 3)))
        cleaned = binary_closing(cleaned, structure=np.ones((5, 5)))
        cleaned = binary_fill_holes(cleaned)
        labels, count = label(cleaned)
        if count:
            sizes = np.bincount(labels.ravel())
            cleaned &= sizes[labels] >= min_component_pixels
        return cleaned.astype(bool)

    def _mask_output(
        self,
        kind: str,
        mask: np.ndarray,
        image: Any,
        store: LocalArtifactStore,
        summary: dict[str, Any],
    ) -> StageOutput:
        with TemporaryDirectory() as temp:
            destination = Path(temp) / "mask.tif"
            self.adapter.write_mask(destination, mask, image)
            artifact = store.put_bytes(kind, destination.read_bytes(), "tif")
        return StageOutput([artifact], {**summary, "water_pixels": int(mask.sum())})

    @staticmethod
    def _read_mask(artifact: Artifact) -> np.ndarray:
        import rasterio

        with rasterio.open(LocalArtifactStore.path(artifact)) as src:
            return src.read(1).astype(bool)

    @staticmethod
    def _polygonize(mask: np.ndarray, image: Any) -> dict[str, Any]:
        features = []
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=image.transform):
            if value == 1:
                features.append(
                    {"type": "Feature", "properties": {"class": "water"}, "geometry": geometry}
                )
        result: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        if image.crs:
            result["crs"] = {"type": "name", "properties": {"name": image.crs}}
        return result

    @staticmethod
    def _statistics(mask: np.ndarray, image: Any) -> dict[str, float | int]:
        pixels = int(mask.sum())
        pixel_area = abs(
            image.transform.a * image.transform.e - image.transform.b * image.transform.d
        )
        # Pixel coordinates and geographic CRS do not carry metre area reliably; report area as unavailable.
        has_metric_area = bool(
            image.crs and "GEOGCS" not in image.crs.upper() and "EPSG:4326" not in image.crs.upper()
        )
        return {
            "water_pixels": pixels,
            "total_pixels": int(mask.size),
            "coverage_fraction": float(pixels / mask.size),
            "pixel_area_native_units": float(pixel_area),
            "water_area_m2": float(pixels * pixel_area) if has_metric_area else 0.0,
        }

    @staticmethod
    def _report(job: Job, stats: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Remote-sensing water extraction report",
                "",
                f"Job: `{job.job_id}`",
                f"Sensor: `{job.request.sensor_type}`",
                f"Water pixels: {stats['water_pixels']}",
                f"Coverage: {stats['coverage_fraction']:.2%}",
                f"Water area (m²): {stats['water_area_m2']:.2f}",
                "",
                "This Phase 1 result uses deterministic processing.",
            ]
        )
