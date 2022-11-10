import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import json
from math import ceil
import random
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from torch.utils.data import Subset

import disstl.datasets.indices  # noqa F401
from disstl.datasets.transforms import Append, Pad, ToDtype, ToTensor
from disstl.datasets.utils import pixel_to_latlon

_logger = logging.getLogger(__name__)

# custom typing hints
Window = Tuple[Tuple[int, int], Tuple[int, int]]
Coord = Tuple[Tuple[float, float], Tuple[float, float]]


def first_random_last_windows(axis_length, window, num_first_last=3, seed=0):
    """Iterates over windows of [first_images, random_middle_images, last_images].

    Args:
        axis_length (int): The length of the axis over which indices will be generated
        window (int): The size of the windows
        num_first_last (int): the number of images to allocate for first_images and last_images

    Yields:
        List(int): list of indices selected for this particular window
    """
    # Seed the random number generator so that multiple workers can generate the same windows
    random.seed(seed)
    indices = list(range(axis_length))
    if len(indices) <= num_first_last * 2:
        yield indices
    else:
        first_indices = indices[:num_first_last]
        last_indices = indices[-num_first_last:]
        middle_indices = indices[num_first_last:-num_first_last]
        middle_step_size = window - 2 * num_first_last

        # Ensure middle_indices is equally divisible by middle_step_size
        for i in range(middle_step_size - (len(middle_indices) % middle_step_size)):
            middle_indices.append(random.choice(middle_indices))

        # Shuffle the middle indices so we create random subsets
        random.shuffle(middle_indices)
        for i in range(0, len(middle_indices), middle_step_size):
            indices = first_indices + sorted(middle_indices[i : i + middle_step_size]) + last_indices
            assert len(indices) == window
            yield indices


def load_transforms(input_shape: List[int], transforms: T.Compose) -> T.Compose:
    transforms = [Pad(tuple(input_shape)), ToTensor(), ToDtype(torch.float), transforms]
    return T.Compose(transforms)


def load_indices(spectral_indices: List[Dict[str, Any]], bands: List[str]) -> T.Compose:
    indices = [instantiate(idx) for idx in spectral_indices]
    for i in range(len(indices)):
        band_args = []
        for j, band in enumerate(bands):
            for b in indices[i].ordered_bands:
                if b in band:
                    band_args.append(j)
        indices[i] = indices[i](*band_args)
    indices = T.Compose([Append(idx, dim=0) for idx in indices])
    return indices


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([i["x"] for i in batch], dim=0)
    images = [i["images"] for i in batch]
    windows = [i["window"] for i in batch]
    temporal_windows = [i["temporal_window"] for i in batch]
    coords = [i["coords"] for i in batch]
    transforms = [i["transform"] for i in batch]
    sources = [i["sources"] for i in batch]
    return dict(
        x=x,
        images=images,
        windows=windows,
        temporal_windows=temporal_windows,
        coords=coords,
        transform=transforms,
        sources=sources,
    )


class EmptyHDF5Error(Exception):
    pass


class InferenceHDF5Dataset(torch.utils.data.Dataset):
    """
    Inference Dataset loaded from HDF5 preprocessed file
    """

    shape: Tuple[int, int]
    images: List[str]
    windows: List[Window]
    temporal_windows: List[int]
    coords: List[Coord]
    transform: np.ndarray

    def __init__(
        self,
        path: str,
        chip_shape: Tuple[int, int],
        transforms: T.Compose,
        stride: Optional[Tuple[int, int]] = None,
        band_names: List[str] = ["red", "green", "blue"],
        temporal_window_size: int = 10,
        temporal_repeats: int = 3,
        temporal_num_first_last: int = 3,
    ):
        self.chip_shape = chip_shape
        self.transforms = transforms
        self.stride = stride
        self.band_names = band_names
        self.path = path
        if len(self.images) == 0:
            raise EmptyHDF5Error("The input HDF5 contains no imagery.")

        # Ensure we don't have any images with different shapes as we currently make assumptions based on this
        assert all([self.get_image_shape(self.images[0]) == self.get_image_shape(image) for image in self.images])
        self.shape = self.get_image_shape(self.images[0])

        # Ensure we don't have any images with different transforms as we currently make assumptions based on this
        assert all([all(self.get_transform(self.images[0]) == self.get_transform(image)) for image in self.images])
        self.transform = self.get_transform(self.images[0])

        self.crs = self.get_crs()

        self.temporal_window_size = temporal_window_size
        self.temporal_repeats = temporal_repeats
        self.temporal_num_first_last = temporal_num_first_last

        _logger.info(f"Using {self.temporal_repeats} temporal repeats during inference")

    @cached_property
    def file(self) -> h5py.File:
        return h5py.File(self.path, "r")

    @cached_property
    def band_indices(self):
        # Get bands from cube metadata
        band_metadata = json.loads(self.file["metadata"]["cube"][0])["bands"]
        return [band_metadata[band] for band in self.band_names]

    @cached_property
    def windows(self) -> List[Window]:
        """Extract coordinates using sliding window generator"""
        rows, cols = self.shape
        chip_rows, chip_cols = self.chip_shape

        if self.stride is None:
            row_div_ceil = ceil(rows / chip_rows)
            col_div_ceil = ceil(cols / chip_cols)
            stride_h, stride_w = ceil(rows / row_div_ceil), ceil(cols / col_div_ceil)
        else:
            stride_h, stride_w = self.stride

        windows = []
        for i in range(0, rows, stride_h):
            for j in range(0, cols, stride_w):
                rowstart, colstart = i, j
                rowend, colend = i + chip_rows, j + chip_cols
                if rowend > rows:
                    rowstart -= rowend - rows
                    rowend = rows
                if colend > cols:
                    colstart -= colend - cols
                    colend = cols
                windows.append(((rowstart, rowend), (colstart, colend)))

        return windows

    @cached_property
    def temporal_windows(self) -> List[int]:
        # TODO: limit number of repeats when we have lots of imagery
        temporal_windows = []
        [
            temporal_windows.extend(
                first_random_last_windows(
                    len(self.images),
                    window=self.temporal_window_size,
                    num_first_last=self.temporal_num_first_last,
                    seed=repeat_idx,
                )
            )
            for repeat_idx in range(self.temporal_repeats)
        ]
        assert all(np.unique(temporal_windows) == range(len(self.images)))
        return temporal_windows

    @cached_property
    def images(self) -> List[str]:
        return sorted(list(self.file["imagery"].keys()))

    @cached_property
    def sources(self):
        return [self.get_image_source(image) for image in self.images]

    @cached_property
    def coords(self) -> List[Coord]:
        return [pixel_to_latlon(w, self.transform) for w in self.windows]

    @classmethod
    def from_indices(cls, indices: List[int], path: str, *args, **kwargs):
        """Subset dataset by indices"""
        dataset = cls(path, *args, **kwargs)
        return Subset(dataset, indices)

    def get_transform(self, image: str) -> np.ndarray:
        return self.file["imagery"][image]["transform"][:].flatten()

    def get_crs(self) -> str:
        return json.loads(self.file["metadata"]["cube"][0])["crs"]

    def get_image_shape(self, image: str) -> Tuple[int, int]:
        return self.file["imagery"][image]["image"].shape[1:]

    def get_image_source(self, image: str) -> List[int]:
        return [source.decode("ascii") for source in self.file["imagery"][image]["stac_item_ids"][:]]

    def load_chip(self, image: str, window: Window) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = window
        return self.file["imagery"][image]["image"][self.band_indices, xmin:xmax, ymin:ymax]

    def __len__(self) -> int:
        return len(self.windows) * len(self.temporal_windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        # Map idx to window indices
        window_idx = idx // len(self.temporal_windows)
        temporal_window_idx = idx % len(self.temporal_windows)

        window = self.windows[window_idx]
        coords = self.coords[window_idx]
        temporal_window = self.temporal_windows[temporal_window_idx]
        images = [self.images[temporal_idx] for temporal_idx in temporal_window]
        chips = [self.load_chip(image=image, window=window) for image in images]
        cube = [self.transforms(chip) for chip in chips]
        sources = [self.sources[temporal_idx] for temporal_idx in temporal_window]
        return dict(
            x=torch.stack(cube, dim=0),
            images=images,
            window=window,
            temporal_window=temporal_window,
            coords=coords,
            transform=self.transform,
            crs=self.crs,
            sources=sources,
        )
