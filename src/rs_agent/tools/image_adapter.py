from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import rasterio
from PIL import Image
from rasterio.env import set_gdal_config
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

# Rasterio wheels include their own projection database. Explicitly registering it
# avoids Windows installations accidentally resolving an unrelated/missing PROJ path.
_RASTERIO_DATA = Path(rasterio.__file__).resolve().parent
os.environ.setdefault("GDAL_DATA", str(_RASTERIO_DATA / "gdal_data"))
os.environ.setdefault("PROJ_DATA", str(_RASTERIO_DATA / "proj_data"))
set_gdal_config("GDAL_DATA", os.environ["GDAL_DATA"])
set_gdal_config("PROJ_DATA", os.environ["PROJ_DATA"])


@dataclass
class ImageData:
    pixels: np.ndarray  # H, W, C; float32
    crs: str | None
    transform: Affine
    source_profile: dict[str, Any]
    source_uri: str

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def bands(self) -> int:
        return int(self.pixels.shape[2])

    def manifest(self) -> dict[str, Any]:
        return {
            "uri": self.source_uri,
            "width": self.width,
            "height": self.height,
            "band_count": self.bands,
            "crs": self.crs,
            "transform": list(self.transform)[:6],
            "pixel_size": [abs(self.transform.a), abs(self.transform.e)],
            "dtype": str(self.pixels.dtype),
        }


class ImageAdapter:
    """Narrow reader/writer interface for Phase 1 inputs and GeoTIFF outputs."""

    RASTER_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".tif", ".tiff"})

    def read(self, uri: str) -> ImageData:
        path = Path(uri.replace("file://", ""))
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() in self.RASTER_EXTENSIONS:
            with rasterio.open(path) as src:
                data = src.read(out_dtype="float32")
                if data.ndim != 3:
                    raise ValueError("Only banded rasters are supported")
                return ImageData(
                    pixels=np.moveaxis(data, 0, -1),
                    crs=src.crs.to_string() if src.crs else None,
                    transform=src.transform,
                    source_profile=src.profile.copy(),
                    source_uri=str(path),
                )
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        return ImageData(
            pixels=pixels,
            crs=None,
            transform=Affine.identity(),
            source_profile={"driver": "PNG" if path.suffix.lower() == ".png" else "JPEG"},
            source_uri=str(path),
        )

    def read_window(self, uri: str, window: Window) -> ImageData:
        """Read a GeoTIFF window for bounded-memory tile processing."""
        path = Path(uri.replace("file://", ""))
        with rasterio.open(path) as src:
            data = src.read(window=window, out_dtype="float32")
            return ImageData(
                pixels=np.moveaxis(data, 0, -1),
                crs=src.crs.to_string() if src.crs else None,
                transform=src.window_transform(window),
                source_profile=src.profile.copy(),
                source_uri=str(path),
            )

    @staticmethod
    def resample(image: ImageData, max_dimension: int) -> ImageData:
        """Downsample an in-memory image while preserving the spatial footprint."""
        if max_dimension < 1:
            raise ValueError("max_dimension must be positive")
        longest = max(image.height, image.width)
        if longest <= max_dimension:
            return image
        scale = max_dimension / longest
        out_height, out_width = round(image.height * scale), round(image.width * scale)
        from scipy.ndimage import zoom

        pixels = zoom(
            image.pixels, (out_height / image.height, out_width / image.width, 1), order=1
        )
        transform = image.transform * Affine.scale(
            image.width / out_width, image.height / out_height
        )
        return ImageData(
            pixels.astype(np.float32), image.crs, transform, image.source_profile, image.source_uri
        )

    @staticmethod
    def reproject_to(image: ImageData, target_crs: str) -> ImageData:
        """Limited in-memory reprojection for explicitly mismatched sample CRSs."""
        if not image.crs:
            raise ValueError("Cannot reproject an image without a CRS")
        if image.crs == target_crs:
            return image
        left, bottom, right, top = rasterio.transform.array_bounds(
            image.height, image.width, image.transform
        )
        transform, width, height = calculate_default_transform(
            image.crs, target_crs, image.width, image.height, left, bottom, right, top
        )
        destination = np.empty((image.bands, height, width), dtype=np.float32)
        source = np.moveaxis(image.pixels, -1, 0)
        for index in range(image.bands):
            reproject(
                source=source[index],
                destination=destination[index],
                src_transform=image.transform,
                src_crs=image.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        return ImageData(
            np.moveaxis(destination, 0, -1),
            target_crs,
            transform,
            image.source_profile,
            image.source_uri,
        )

    @staticmethod
    def normalize(pixels: np.ndarray) -> np.ndarray:
        result = np.empty_like(pixels, dtype=np.float32)
        for index in range(pixels.shape[-1]):
            band = pixels[..., index]
            low, high = np.nanpercentile(band, (2, 98))
            result[..., index] = 0 if high <= low else np.clip((band - low) / (high - low), 0, 1)
        return result

    def write_mask(self, path: str | Path, mask: np.ndarray, reference: ImageData) -> None:
        profile = {
            "driver": "GTiff",
            "height": reference.height,
            "width": reference.width,
            "count": 1,
            "dtype": "uint8",
            "transform": reference.transform,
            "compress": "lzw",
            "nodata": 255,
        }
        if reference.crs:
            profile["crs"] = reference.crs
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
