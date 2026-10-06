"""Fig. 2: CHM, predicted conservation value, ground truth and correctness map for one 1 m tile.

Paper examples: 636160_6689280_1280.tif and 506880_6268160_1280.tif.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer
import rasterio

from chmcons.data import normalize, read_tile
from chmcons.model import load_unet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tile", help="GeoTIFF tile produced by make_tiles.py")
parser.add_argument("--checkpoint", default="runs/1m/model.pth")
parser.add_argument("--out", default="results")
args = parser.parse_args()

chm, labels = read_tile(args.tile)
with rasterio.open(args.tile) as src:
    x, y = src.xy(src.height // 2, src.width // 2)
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)
print(f"tile center: {lat:.6f}N, {lon:.6f}E")

model = load_unet(args.checkpoint, chm.shape[0])
with torch.no_grad():
    pred = model(torch.from_numpy(normalize(chm))[None, None]).squeeze().numpy()

truth = np.where(labels == -1, np.nan, labels)
correct = np.where(labels == -1, np.nan, (pred > 0.5) == (labels == 1))
accuracy = np.nanmean(correct)

fig, axs = plt.subplots(1, 4, figsize=(12, 24), constrained_layout=True)
im = axs[0].imshow(chm, cmap="grey_r")
axs[0].set_title("CHM (input)")
plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
im = axs[1].imshow(pred, cmap="PiYG", vmin=0, vmax=1)
axs[1].set_title("Predicted naturalness (output)")
plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
axs[2].imshow(chm, cmap="grey_r", alpha=0.6)
axs[2].imshow(truth, cmap="PiYG", alpha=0.6)
axs[2].set_title("Ground truth")
axs[3].imshow(correct, cmap=ListedColormap(["darkblue", "gold"]))
axs[3].set_title(f"Evaluation (accuracy = {accuracy:.4f})")
axs[3].legend(handles=[Patch(color="gold", label="Correct"), Patch(color="darkblue", label="Incorrect")],
              loc="upper left")
for ax in axs:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

Path(args.out).mkdir(parents=True, exist_ok=True)
fig.savefig(Path(args.out) / f"inference_results_{Path(args.tile).stem}.pdf", bbox_inches="tight")
