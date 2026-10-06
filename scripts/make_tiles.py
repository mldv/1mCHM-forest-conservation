"""Build the labelled 1280x1280 m GeoTIFF tiles (band 1: CHM in dm, band 2: 0 background / 1 low / 2 high value).

The CHM is read through skogdata from its cache folder (CACHE in .env) and missing files are downloaded from
Skogsstyrelsen's FTP server, unless SKOGDATA_OFFLINE=1. The CHM version is fixed to what was used in the paper.
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

CHM_METADATA = "20250131"
CHM_CUTOFF = "2025-01-31"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--labels", default="data/labels.gpkg.zip")
parser.add_argument("--grid", default="data/tiles.gpkg")
parser.add_argument("--out", default="tiles")
parser.add_argument("--splits", nargs="+", default=["training", "validation", "test"])
parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="only tiles inside this box (EPSG:3006)")
parser.add_argument("--list-files", metavar="FILE",
                    help="only write the FTP paths of the CHM files needed (e.g. for an FTP client) to FILE")
args = parser.parse_args()

chm_source = DataSourceCatalog.Tradhojd
chm_source.configure(metadata=CHM_METADATA, cutoff=CHM_CUTOFF)

grid = geopandas.read_file(args.grid)
grid = grid[grid["split"].isin(args.splits)]
if args.bbox:
    grid = grid[grid.within(shapely.box(*args.bbox))]

if args.list_files:
    files = {str(f) for tile in grid.geometry for f in chm_source.required_files(tile)}
    files = [str(f) for f in chm_source.metadata_source.files] + sorted(files)
    Path(args.list_files).write_text("\n".join(files) + "\n")
    print(f"{len(files)} files listed in {args.list_files}")
    raise SystemExit

labels = geopandas.read_file(args.labels).set_index("naturalness").geometry
low, high = labels[0], labels[1]
shapely.prepare(low)
shapely.prepare(high)

failed = []
for tile, split in tqdm(zip(grid.geometry, grid["split"]), total=len(grid), desc="tiles"):
    x0, y0, *_ = tile.bounds
    path = Path(args.out) / split / f"{int(x0)}_{int(y0)}_1280.tif"
    if path.exists():
        continue

    try:
        chm, transform = chm_source(tile, padding=0)
    except FileNotFoundError as e:
        failed.append(f"{path.name}: {e}")
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

    path.parent.mkdir(parents=True, exist_ok=True)
    raster = np.stack([chm, lab])
    with rasterio.open(
        path, "w", driver="GTiff", compress="lzw", crs="EPSG:3006", transform=transform,
        count=2, height=raster.shape[1], width=raster.shape[2], dtype=raster.dtype,
    ) as dst:
        dst.write(raster)

if failed:
    print(f"{len(failed)} tiles could not be built:", *failed, sep="\n")
