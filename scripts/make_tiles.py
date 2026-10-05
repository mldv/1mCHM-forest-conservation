"""Build labelled 1280x1280 m GeoTIFF tiles (band 1: CHM in dm, band 2: 0 background / 1 low / 2 high value).

The CHM is downloaded on demand from Skogsstyrelsen via skogdata (cache folder set by CACHE in .env).
Tiles without CHM coverage or without any label are skipped.
"""

import argparse
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import rasterio.features
import shapely
from skogdata import DataSourceCatalog
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--labels", default="data/labels.gpkg.zip")
parser.add_argument("--grid", default="data/tiles.gpkg")
parser.add_argument("--out", default="tiles")
parser.add_argument("--splits", nargs="+", default=["training", "validation", "test"])
args = parser.parse_args()

labels = geopandas.read_file(args.labels).set_index("naturalness").geometry
low, high = labels[0], labels[1]
shapely.prepare(low)
shapely.prepare(high)

grid = geopandas.read_file(args.grid)
grid = grid[grid["split"].isin(args.splits)]

for tile, split in tqdm(zip(grid.geometry, grid["split"]), total=len(grid)):
    x0, y0, *_ = tile.bounds
    path = Path(args.out) / split / f"{int(x0)}_{int(y0)}_1280.tif"
    if path.exists():
        continue

    try:
        chm, transform = DataSourceCatalog.Tradhojd(tile, padding=0)
    except (FileNotFoundError, AssertionError):
        continue
    chm = chm[0]

    low_t = shapely.clip_by_rect(low, *tile.bounds)
    high_t = shapely.clip_by_rect(high, *tile.bounds)
    both = high_t.intersection(low_t)

    lab = np.zeros_like(chm)
    for value, geom in ((1, low_t.difference(both)), (2, high_t.difference(both))):
        if geom.is_empty or geom.area == 0:
            continue
        mask = rasterio.features.rasterize([geom], out_shape=chm.shape, transform=transform)
        lab += mask * value * (lab == 0)
    if lab.max() == 0:
        continue

    path.parent.mkdir(parents=True, exist_ok=True)
    raster = np.stack([chm, lab])
    with rasterio.open(
        path, "w", driver="GTiff", compress="lzw", crs="EPSG:3006", transform=transform,
        count=2, height=raster.shape[1], width=raster.shape[2], dtype=raster.dtype,
    ) as dst:
        dst.write(raster)
