"""
Train the U-Net at one resolution with the combined loss; keep the checkpoint with the lowest validation loss.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chmcons.data import TILE_M, H5Dataset, subset
from chmcons.losses import combined_loss
from chmcons.metrics import Confusion
from chmcons.model import build_unet

# (batch size, learning rate) used for the published models, per resolution in m
HPARAMS = {
    1: (2, 0.001),
    2: (6, 0.005),
    5: (1024, 0.016),
    8: (1024, 0.016),
    10: (1024, 0.016),
    20: (1024, 0.016),
    40: (32, 0.001),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resolution", type=int, required=True, choices=sorted(HPARAMS))
parser.add_argument("--h5", default="h5")
parser.add_argument("--out", default=None, help="output folder (default: runs/<resolution>m)")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--alpha", type=float, default=0.3, help="focal loss weight")
parser.add_argument("--beta", type=float, default=0.4, help="pixel accuracy loss weight")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--min-delta", type=float, default=1e-4)
parser.add_argument("--fraction", type=float, default=1.0, help="fraction of tiles used (loss grid search)")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=None, help="random seed for weights and shuffling")
args = parser.parse_args()

batch_size, lr = HPARAMS[args.resolution]
batch_size = args.batch_size or batch_size
lr = args.lr or lr
out = Path(args.out or f"runs/{args.resolution}m")
out.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seed is not None:
    torch.manual_seed(args.seed)

datasets = {}
for split in ("training", "validation"):
    ds = H5Dataset(Path(args.h5) / f"{split}-{args.resolution}m.h5")
    datasets[split] = subset(ds, args.fraction) if args.fraction < 1 else ds
loaders = {
    split: DataLoader(ds, batch_size=batch_size, shuffle=split == "training", num_workers=args.workers,
                      pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    for split, ds in datasets.items()
}

net = build_unet(TILE_M // args.resolution)
model = torch.nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def run_epoch(split: str) -> tuple[float, float]:
    training = split == "training"
    model.train(training)
    total_loss, conf = 0.0, Confusion()
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loaders[split], desc=split, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x)
            loss = combined_loss(pred, y, args.alpha, args.beta)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            conf.update(pred.detach(), y)
    return total_loss / len(loaders[split]), conf.summary()["accuracy"]


best_loss = stop_loss = float("inf")
stale = 0
with open(out / "log.csv", "w", newline="") as log:
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch("training")
        val_loss, val_acc = run_epoch("validation")
        writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])
        log.flush()
        print(f"epoch {epoch}: loss {train_loss:.4f}/{val_loss:.4f}  accuracy {train_acc:.4f}/{val_acc:.4f}")

        if val_loss < stop_loss - args.min_delta:
            stop_loss, stale = val_loss, 0
        else:
            stale += 1
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "val_accuracy": val_acc, "args": vars(args) | {"batch_size": batch_size, "lr": lr}},
                       out / "model.pth")
        if stale >= args.patience:
            print(f"early stopping after epoch {epoch}")
            break
