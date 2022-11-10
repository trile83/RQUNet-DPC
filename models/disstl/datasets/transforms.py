from typing import Dict, List, Optional, Tuple, Union

import kornia.augmentation as K
import numpy as np
import rasterio
import torch
import torch.nn as nn
from functools import cached_property

from disstl.imagery.util import image_enhancement as ime

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_vrt(vrt: str, window: Tuple[Tuple[int, int]]) -> np.ndarray:
    """Load a window from a vrt"""
    with rasterio.open(vrt) as dataset:
        x = dataset.read(window=window, boundless=True, fill_value=0)

    return x


class ConvertBHWCtoBCHW(nn.Module):
    """
    Convert tensor from (B, H, W, C) to (B, C, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class LoadVRT(object):
    """
    Load a VRT and extract a chip
    """

    def __call__(self, x: Tuple[str, Tuple[Tuple[int, int]]]) -> np.ndarray:
        vrt, window = x
        return load_vrt(vrt, window)


class FillMissingPixels(object):
    """
    Fill bad pixels indicated by fill_value with fill_with
    """

    def __init__(self, fill_value: float, fill_with: float):
        self.fill_value = fill_value
        self.fill_with = fill_with

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x[x == self.fill_value] = self.fill_with
        return x


# Transforms
class DiSSTLTransform:
    # This class exists to give a sane str repr to transforms so we may hash them for comparisons between training runs
    def __str__(self):
        strrep = "".join([str(v).split("object at")[0] for k, v in dict(sorted(self.__dict__.items())).items()])
        return strrep


class SelectBands(DiSSTLTransform):
    """
    Select specific bands from multispectral chip
    """

    def __init__(self, indices: str):
        self.indices = indices

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if x.ndim == 3:
            return x[self.indices, ...]
        elif x.ndim == 4:
            return x[:, self.indices, ...]
        else:
            raise ValueError("inputs to SelectBands should have either 3 or 4 dimensions")


class ClipBands(DiSSTLTransform):
    """
    Clip Bands in temporal stack of chips to per-band [min, max] ranges
    """

    def __init__(self, mins: List[float], maxs: List[float]):
        self._mins = mins
        self._maxs = maxs

    @cached_property
    def mins(self):
        if self.xdim == 3:
            mins = torch.Tensor(self._mins)[:, None, None]
        if self.xdim == 4:
            mins = torch.Tensor(self._mins)[None, :, None, None]
        return mins

    @cached_property
    def maxs(self):
        if self.xdim == 3:
            maxs = torch.Tensor(self._maxs)[:, None, None]
        if self.xdim == 4:
            maxs = torch.Tensor(self._maxs)[None, :, None, None]
        return maxs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = torch.where(x < self.mins, self.mins, x)
            x = torch.where(x > self.maxs, self.maxs, x)
            return x
        except AttributeError:
            # self.mins could not resolve because it needs self.xdim, set it and try again.
            # This happens exactly once, the first time through
            self.xdim = x.ndim
            return self(x)


class MinMaxNormalize(DiSSTLTransform):
    """
    Generic min/max normalize preprocessor
    """

    def __init__(self, mins: List[float], maxs: List[float]):
        self._mins = mins
        self._maxs = maxs

    @cached_property
    def mins(self):
        if self.xdim == 3:
            mins = torch.Tensor(self._mins)[:, None, None]
        if self.xdim == 4:
            mins = torch.Tensor(self._mins)[None, :, None, None]
        return mins

    @cached_property
    def maxs(self):
        if self.xdim == 3:
            maxs = torch.Tensor(self._maxs)[:, None, None]
        if self.xdim == 4:
            maxs = torch.Tensor(self._maxs)[None, :, None, None]
        return maxs

    @cached_property
    def denom(self):
        return self.maxs - self.mins

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return (x - self.mins) / self.denom
        except AttributeError:
            # self.mins could not resolve because it needs self.xdim, set it and try again
            # This happens exactly once, the first time through
            self.xdim = x.ndim
            return self(x)


class HistEqualize(DiSSTLTransform):
    """
    Histogram equalization
    """

    def __init__(self, colorspace: str = "hsv"):
        self.colorspace = colorspace
        self.equalizer = ime.CDFEqualizer()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = ime.norm_array(x.numpy(), scale=255.0)
        x = self.equalizer.equalize(x.astype(np.uint8), colorspace=self.colorspace)
        x = x.astype(np.float32) / 255.0  # range[0, 1]
        return torch.from_numpy(x)


class ToTensor(DiSSTLTransform):
    """
    Convert numpy array to tensor
    """

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


class ToDtype(DiSSTLTransform):
    """
    Convert torch tensor to dtype
    """

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)


class AsDtype(DiSSTLTransform):
    """
    Convert torch tensor to dtype
    """

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.type(self.dtype)


class Pad(DiSSTLTransform):
    """
    Pad a numpy array to a specific shape.
    """

    def __init__(self, shape: Tuple[int, int], orientation: Optional[str] = "botright"):
        self.shape = shape
        self.orientation = orientation

    def __call__(self, x: np.ndarray, pad_value: int = 0) -> np.ndarray:
        if x.shape[1:] != self.shape:
            if len(x.shape) > 2:
                if self.orientation == "center":
                    diff = self.shape[0] - x.shape[1]
                    top = int(diff / 2)
                    bot = diff - top
                    diff = self.shape[1] - x.shape[2]
                    left = int(diff / 2)
                    right = diff - left
                else:  # "botright":
                    top, bot = 0, self.shape[0] - x.shape[1]
                    left, right = 0, self.shape[1] - x.shape[2]
                padding = [(0, 0), (top, bot), (left, right)]

            else:
                if self.orientation == "center":
                    diff = self.shape[0] - x.shape[0]
                    top = int(diff / 2)
                    bot = diff - top
                    diff = self.shape[1] - x.shape[1]
                    left = int(diff / 2)
                    right = diff - left
                else:  # "botright":
                    top, bot = 0, self.shape[0] - x.shape[0]
                    left, right = 0, self.shape[1] - x.shape[1]
                padding = [(top, bot), (left, right)]

            x = np.pad(x, padding, mode="constant", constant_values=pad_value)
        return x


# Augmentations


class Append(nn.Module):
    """Append an additional channel."""

    def __init__(self, transform: nn.Module, dim: int = 0) -> None:
        """Initialize a new transform instance.

        Args:
            dim: dimension of channels in the input torch.Tensors (default: 0)
        """
        super().__init__()
        self.dim = dim
        self.transform = transform

    def forward(
        self, sample: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Create a band and append to image channels.

        Args:
            sample: a single data sample

        Returns:
            a sample where the image has an additional channel
        """
        if isinstance(sample, dict):  # Batch of Temporal Stacks
            image = sample["x"]
            # TODO Check that forward methods of indices.py will point to band dimension instead of time dimension
            channel = self.transform(image)
            channel = channel.unsqueeze(self.dim)
            sample["x"] = torch.cat([image, channel], dim=self.dim)
        elif sample.ndim == 4:  # Temporal Stack (t x band x h x w)
            image = sample
            channel = self.transform(image)
            channel = channel.unsqueeze(1)
            sample = torch.cat([image, channel], dim=1)
        else:  # Single Multi-band chip (band x h x w)
            image = sample.unsqueeze(0)
            channel = self.transform(image)
            channel = channel.unsqueeze(0)
            sample = torch.cat([image, channel], dim=self.dim + 1)
            sample = sample.squeeze(0)

        return sample


class AugmentationSequential(nn.Module):
    """Wrapper around kornia AugmentationSequential to handle input dicts."""

    def __init__(self, *args: nn.Module, data_keys: List[str]) -> None:
        """Initialize a new augmentation sequential instance.
        Args:
            *args: Sequence of kornia augmentations
            data_keys: List of inputs to augment (e.g. ["image", "mask", "boxes"])
        """
        super().__init__()

        order = {}
        self.mask_keys = []
        self.image_keys = []
        for key in data_keys:
            if key == "x":
                order[key] = "input"
                self.image_keys.append(key)
            elif key == "boxes":
                order[key] = "bbox"
                self.image_keys.append(key)
            elif key.find("mask") > -1:
                self.mask_keys.append(key)
                order[key] = "mask"
            else:
                # https://kornia.readthedocs.io/en/latest/augmentation.container.html#augmentation-sequential
                raise NotImplementedError(f"{key} not implemented for AugmentationSequential")
        self.input_keys = self.image_keys + self.mask_keys
        self.augs = K.AugmentationSequential(*args, data_keys=[order[key] for key in self.input_keys])

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform augmentations and update data dict.
        Args:
            sample: the input
        Returns:
            the augmented input
        """

        inputs = [sample[k] for k in self.image_keys]
        if len(self.mask_keys) == 0:  # There are no masks in this sample
            outputs_list: Union[torch.Tensor, List[torch.Tensor]] = self.augs(*inputs)
            outputs_list = outputs_list if isinstance(outputs_list, list) else [outputs_list]
            outputs: Dict[str, torch.Tensor] = {k: v for k, v in zip(self.image_keys, outputs_list)}

        else:
            x_dtype = sample["x"].dtype
            mask_dtype = sample[self.mask_keys[0]].dtype
            # Incoming masks do not have channels, but mulitspectral images do,
            # so expand a channel dimension for the masks (unsqueeze at index -3)
            mask_inputs = [sample[k].unsqueeze(-3).type(x_dtype) for k in self.mask_keys]
            inputs.extend(mask_inputs)
            outputs_list: Union[torch.Tensor, List[torch.Tensor]] = self.augs(*inputs)
            outputs_list = outputs_list if isinstance(outputs_list, list) else [outputs_list]
            outputs: Dict[str, torch.Tensor] = {
                k: (v if k not in self.mask_keys else v.squeeze(dim=-3).type(mask_dtype))
                for k, v in zip(self.input_keys, outputs_list)
            }

        sample.update(outputs)
        return sample
