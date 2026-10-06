"""
U-Net mapping a single-channel CHM tile to a per-pixel probability of high conservation value,
with feature widths scaled to the tile size.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-level U-Net mapping a 1-channel CHM tile to a per-pixel probability of high conservation value."""

    def __init__(self, features: list[int], in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for f in features:
            self.downs.append(_double_conv(channels, f))
            channels = f

        self.bottleneck = _double_conv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(_double_conv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i, skip in enumerate(reversed(skips)):
            x = self.ups[2 * i](x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = self.ups[2 * i + 1](torch.cat((skip, x), dim=1))

        return torch.sigmoid(self.final_conv(x))


def build_unet(tile_px: int) -> UNet:
    """U-Net whose feature widths scale with the tile side, e.g. 1280 px -> [80, 160, 320, 640]."""
    return UNet(features=[tile_px // 2**i for i in (4, 3, 2, 1)])


def load_unet(path: str, tile_px: int, device: str | torch.device = "cpu") -> UNet:
    model = build_unet(tile_px)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.to(device).eval()
