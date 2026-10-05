"""Pack the GeoTIFF tiles of one split into an HDF5 file at a given resolution (m), e.g. h5/training-1m.h5."""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from chmcons.data import TILE_M, downsample, read_tile


def load(path: Path, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    return downsample(*read_tile(path), resolution)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("split", choices=["training", "validation", "test"])
parser.add_argument("--resolution", type=int, default=1, help="pixel size in m (1, 2, 5, 8, 10, 20, 40)")
parser.add_argument("--tiles", default="tiles")
parser.add_argument("--out", default="h5")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

paths = sorted((Path(args.tiles) / args.split).glob("*.tif"))
n = TILE_M // args.resolution
out = Path(args.out) / f"{args.split}-{args.resolution}m.h5"
out.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(out, "w") as f, Pool(args.workers) as pool:
    opts = dict(dtype=np.float32, compression="gzip", compression_opts=4)
    chm_ds = f.create_dataset("chm", (len(paths), 1, n, n), chunks=(1, 1, n, n), **opts)
    lab_ds = f.create_dataset("labels", (len(paths), n, n), chunks=(1, n, n), **opts)
    f["names"] = [p.name for p in paths]
    for i, (chm, lab) in enumerate(tqdm(pool.imap(partial(load, resolution=args.resolution), paths), total=len(paths))):
        chm_ds[i, 0] = chm
        lab_ds[i] = lab
