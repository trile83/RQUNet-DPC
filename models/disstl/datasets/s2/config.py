from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import kornia.augmentation as K
import torchvision.transforms as T

from disstl.datasets.transforms import Pad, ToTensor, ToDtype, SelectBands, ClipBands, HistEqualize


# Note these are 0-indexed and not based on namings like B01, B02, etc..
BANDS = ["Blue", "Green", "Red"]
IDX2BAND = OrderedDict({i: b for i, b in enumerate(BANDS)})
BAND2IDX = OrderedDict({b: i for i, b in enumerate(BANDS)})

BAND_INDICES = [0, 1, 2]

DRA_INTERVAL = [2, 98]

BAND_STATS = OrderedDict(
    {
        "Blue": {"min": 0.0, "max": 26841.0, "mean": 2143.208547, "std": 1759.259314},
        "Green": {"min": 0.0, "max": 23912.0, "mean": 2171.483802, "std": 1690.687129},
        "Red": {"min": 0.0, "max": 21684.0, "mean": 2159.983605, "std": 1680.718736},
    }
)

BAND_MIN = [BAND_STATS[b]["min"] for b in BANDS]
BAND_MAX = [BAND_STATS[b]["max"] for b in BANDS]
BAND_MEAN = [BAND_STATS[b]["mean"] for b in BANDS]
BAND_STD = [BAND_STATS[b]["std"] for b in BANDS]


def load_transforms(
    input_shape: List[int],
    bands: Optional[List[int]] = None,
    mins: List[float] = BAND_MIN,
    maxs: List[float] = BAND_MAX,
    colorspace: str = "hsv",
) -> T.Compose:

    transforms = [
        Pad(tuple(input_shape)),
        ToTensor(),
        ToDtype(torch.float32),
        SelectBands(BAND_INDICES),
        ClipBands(mins, maxs),
        HistEqualize(colorspace=colorspace),
    ]

    if bands is not None:
        transforms.append(SelectBands(bands))

    return T.Compose(transforms)


def augment_transforms(input_shape: Tuple[int, int], per_frame: bool = False) -> nn.Sequential:

    augs = [
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomAffine(degrees=360, resample="bilinear", p=0.5),
        K.RandomResizedCrop(size=input_shape, scale=(0.8, 1.0), ratio=(0.75, 1.33), resample="bilinear", p=0.5),
    ]

    augs = K.VideoSequential(*augs, data_format="BTCHW", same_on_frame=not per_frame)
    return augs
