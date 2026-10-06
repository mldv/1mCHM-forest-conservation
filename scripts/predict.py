"""
Predict a conservation value map for any bounding box in Sweden (SWEREF99 TM, EPSG:3006).

The CHM is downloaded via skogdata, cut into 1280x1280 m windows (zero-padded at the far edges)
and written as a float32 GeoTIFF of per-pixel probabilities of high conservation value.
Without --checkpoint, the published model for the resolution is downloaded once (from the GitHub Release)
into checkpoints/.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
import torch
from affine import Affine
from skogdata import DataSourceCatalog

from chmcons.data import TILE_M, normalize
from chmcons.model import load_unet

# Published models, per resolution (m): URL of the GitHub Release asset and its SHA-256
PRETRAINED = {
    2: (
        "https://github.com/mldv/1mCHM-forest-conservation/releases/download/v1.0/chmcons-2m.pth",
        "aa46c9dc2a830fe2f712699c7dc7646a02cb906afabab63bcaabc43576475e96",
    ),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
parser.add_argument("--checkpoint", default=None, help="model.pth from train.py (default: published model)")
parser.add_argument("--resolution", type=int, default=2, help="pixel size in m the model was trained for")
parser.add_argument("--out", default="results/prediction.tif")
parser.add_argument("--png", action="store_true", help="also save a CHM / prediction preview")
args = parser.parse_args()

checkpoint = args.checkpoint
if checkpoint is None:
    if args.resolution not in PRETRAINED:
        parser.error(f"no published model at {args.resolution} m (available: {sorted(PRETRAINED)}), use --checkpoint")
    url, sha256 = PRETRAINED[args.resolution]
    checkpoint = Path("checkpoints") / Path(url).name
    if not checkpoint.exists():
        checkpoint.parent.mkdir(exist_ok=True)
        torch.hub.download_url_to_file(url, str(checkpoint), hash_prefix=sha256)

chm, transform = DataSourceCatalog.Tradhojd(shapely.box(*args.bbox), padding=0)
chm = chm[0].astype(np.float32)

r = args.resolution
h, w = chm.shape[0] // r, chm.shape[1] // r
chm = chm[: h * r, : w * r].reshape(h, r, w, r).mean(axis=(1, 3))
transform = transform * Affine.scale(r)

n = TILE_M // r
padded = np.zeros((math.ceil(h / n) * n, math.ceil(w / n) * n), dtype=np.float32)
padded[:h, :w] = chm
pred = np.zeros_like(padded)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_unet(checkpoint, n, device)
with torch.no_grad():
    for i in range(0, padded.shape[0], n):
        for j in range(0, padded.shape[1], n):
            x = torch.from_numpy(normalize(padded[i : i + n, j : j + n]))[None, None].to(device)
            pred[i : i + n, j : j + n] = model(x).squeeze().cpu().numpy()
pred = pred[:h, :w]

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(out, "w", driver="GTiff", compress="lzw", crs="EPSG:3006", transform=transform,
                   count=1, height=h, width=w, dtype="float32") as dst:
    dst.write(pred, 1)

if args.png:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    plt.colorbar(axs[0].imshow(chm, cmap="grey_r"), ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("CHM (input)")
    plt.colorbar(axs[1].imshow(pred, cmap="PiYG", vmin=0, vmax=1), ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title("Predicted naturalness (output)")
    for ax in axs:
        ax.set_axis_off()
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
