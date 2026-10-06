"""
Evaluate trained models on the test set (Table 1) and write results/results.csv.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chmcons.data import TILE_M, H5Dataset
from chmcons.metrics import Confusion
from chmcons.model import load_unet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resolutions", type=int, nargs="+", default=[40, 20, 10, 8, 5, 2, 1])
parser.add_argument("--runs", default="runs", help="folder with <resolution>m/model.pth")
parser.add_argument("--h5", default="h5")
parser.add_argument("--out", default="results/results.csv")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for res in args.resolutions:
    px = TILE_M // res
    model = load_unet(Path(args.runs) / f"{res}m" / "model.pth", px, device)
    conf = Confusion()
    with torch.no_grad():
        for x, y in tqdm(DataLoader(H5Dataset(Path(args.h5) / f"test-{res}m.h5"), batch_size=args.batch_size),
                         desc=f"{res} m"):
            conf.update(model(x.to(device)), y.to(device))
    rows.append({"resolution_m": res, "tile_px": px} | conf.summary())
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
