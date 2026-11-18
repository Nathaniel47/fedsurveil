"""Small image classifiers. Group norm everywhere: batch norm breaks under non-IID data and DP."""

import torch
from torch import nn


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            _norm(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            _norm(2 * width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1),
            _norm(4 * width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = _norm(cout)
        self.skip = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), _norm(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class SmallResNet(nn.Module):
    """Three stages of one residual block each; used for the 64 px IP102 images."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False), _norm(width), nn.ReLU()
        )
        self.stages = nn.Sequential(
            _Block(width, width, 1),
            _Block(width, 2 * width, 2),
            _Block(2 * width, 4 * width, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stages(self.stem(x))).flatten(1))


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    if name == "cnn":
        return SmallCNN(in_channels, num_classes)
    if name == "resnet":
        return SmallResNet(in_channels, num_classes)
    raise ValueError(f"unknown model {name!r}")


def default_model_name(dataset: str) -> str:
    return "resnet" if dataset == "ip102" else "cnn"
