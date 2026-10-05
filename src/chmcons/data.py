from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

TILE_M = 1280  # tile side on the ground (m), equal to pixels at native 1 m resolution
H_MAX = 300.0  # CHM clipping threshold in decimeters (30 m)
BACKGROUND = -1


def read_tile(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-band GeoTIFF tile: CHM (dm) and labels (0 background, 1 low, 2 high).

    Returns the CHM as float32 and labels as int32 in {-1 background, 0 low, 1 high}.
    """
    with rasterio.open(path) as src:
        chm = src.read(1).astype(np.float32)
        labels = src.read(2).astype(np.int32) - 1
    return chm, labels


def downsample(chm: np.ndarray, labels: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-average the CHM and majority-vote the labels over resolution x resolution blocks.

    Ties in the vote go to the smallest class value (as scipy.stats.mode).
    """
    n = chm.shape[0] // resolution
    chm = chm.reshape(n, resolution, n, resolution).mean(axis=(1, 3))
    blocks = labels.reshape(n, resolution, n, resolution)
    counts = np.stack([(blocks == c).sum(axis=(1, 3)) for c in (-1, 0, 1)])
    labels = counts.argmax(axis=0).astype(np.int32) - 1
    return chm, labels


def normalize(chm: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return (chm / H_MAX).clip(0.0, 1.0)


class H5Dataset(Dataset):
    """Tiles stored by scripts/build_h5.py: `chm` (N, 1, H, W) and `labels` (N, H, W), float32."""

    def __init__(self, path: str | Path, indices: np.ndarray | None = None):
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            self.names = f["names"].asstr()[:]
        self.indices = np.arange(len(self.names)) if indices is None else np.sort(indices)
        self._file = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        j = self.indices[i]
        chm = torch.from_numpy(self._file["chm"][j])
        labels = torch.from_numpy(self._file["labels"][j])
        return normalize(chm), labels


def subset(dataset: H5Dataset, fraction: float, seed: int = 42) -> H5Dataset:
    """Random fraction of the tiles (used for the loss-function grid search)."""
    n = len(dataset)
    np.random.seed(seed)
    return H5Dataset(dataset.path, np.random.choice(n, size=int(n * fraction), replace=False))
