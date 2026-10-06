"""
Forest mask from the Swedish national land cover data (Nationella Marktäckedata, NMD2023, Naturvårdsverket, CC0).

NMD is read for a bounding box through Naturvårdsverket's WMS. The service returns the classes as a
colour-mapped PNG (lossless, unlike its 8-bit GeoTIFF output, which merges similar colours); colours are
mapped back to NMD classes through the legend of the layer.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import json
import math
import urllib.parse
import urllib.request
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject

NMD_WMS = "https://geodata.naturvardsverket.se/inspire/lc-nmd/ows"
NMD_LAYER = "LC.LandCoverRaster.Bas.2023.v2.x"
NMD_RES = 10  # m
MAX_PX = 2000  # pixels per WMS request side

# Forest on firm ground and on wetland (111-128, incl. temporarily non-forest, i.e. clear-cuts), low mountain forest
FOREST_CLASSES = [*range(111, 119), *range(121, 129), 23, 43]


def _get(params: dict) -> bytes:
    url = f"{NMD_WMS}?{urllib.parse.urlencode({'service': 'WMS', 'version': '1.3.0', **params})}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _legend() -> dict[str, int]:
    """Colour (#RRGGBB) -> NMD class."""
    legend = json.loads(_get({"request": "GetLegendGraphic", "format": "application/json", "layer": NMD_LAYER}))
    entries = legend["Legend"][0]["rules"][0]["symbolizers"][0]["Raster"]["colormap"]["entries"]
    return {e["color"].upper(): int(e["quantity"]) for e in entries}


def nmd_classes(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[np.ndarray, Affine]:
    """NMD classes (0 where unknown) on the 10 m NMD grid covering the box (EPSG:3006)."""
    xmin, ymin = math.floor(xmin / NMD_RES) * NMD_RES, math.floor(ymin / NMD_RES) * NMD_RES
    xmax, ymax = math.ceil(xmax / NMD_RES) * NMD_RES, math.ceil(ymax / NMD_RES) * NMD_RES
    w, h = (xmax - xmin) // NMD_RES, (ymax - ymin) // NMD_RES
    legend = _legend()
    classes = np.zeros((h, w), dtype=np.int16)
    for i in range(0, h, MAX_PX):
        for j in range(0, w, MAX_PX):
            nh, nw = min(MAX_PX, h - i), min(MAX_PX, w - j)
            x0, y1 = xmin + j * NMD_RES, ymax - i * NMD_RES
            image = _get({
                "request": "GetMap", "layers": NMD_LAYER, "styles": "", "format": "image/png",
                "crs": "EPSG:3006", "bbox": f"{y1 - nh * NMD_RES},{x0},{y1},{x0 + nw * NMD_RES}",
                "width": nw, "height": nh,
            })
            with warnings.catch_warnings(), rasterio.MemoryFile(image) as f:
                warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
                with f.open() as src:
                    bands = src.read()
                    colormap = src.colormap(1) if src.count == 1 else None
            if colormap is not None:
                lookup = np.zeros(256, dtype=np.int16)
                for k, rgba in colormap.items():
                    lookup[k] = legend.get("#%02X%02X%02X" % rgba[:3], 0)
                block = lookup[bands[0]]
            else:
                rgb = bands[0].astype(np.int32) << 16 | bands[1].astype(np.int32) << 8 | bands[2]
                block = np.zeros(rgb.shape, dtype=np.int16)
                for colour, value in legend.items():
                    block[rgb == int(colour[1:], 16)] = value
            classes[i : i + nh, j : j + nw] = block
    return classes, Affine(NMD_RES, 0, xmin, 0, -NMD_RES, ymax)


def forest_mask(shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """True where NMD classifies the land as forest, on the grid given by shape and transform."""
    h, w = shape
    xmin, ymax = transform * (0, 0)
    xmax, ymin = transform * (w, h)
    classes, nmd_transform = nmd_classes(xmin, ymin, xmax, ymax)
    forest = np.isin(classes, FOREST_CLASSES).astype(np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    reproject(forest, out, src_transform=nmd_transform, src_crs="EPSG:3006", dst_transform=transform,
              dst_crs="EPSG:3006", resampling=Resampling.nearest)
    return out.astype(bool)
