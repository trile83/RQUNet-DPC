import random
from typing import Tuple, List, Dict

import rasterio
import h5py
import torch
import numpy as np
import torchvision.transforms as T

from disstl.datasets.utils import get_latlon_coords, get_random_coords
from disstl.datasets.transforms import load_vrt


def collate_fn(batch) -> Dict:
    x = torch.stack([i["x"] for i in batch], dim=0)
    y = torch.stack([i["y"] for i in batch], dim=0)
    files = [i["files"] for i in batch]
    coords = [i["coords"] for i in batch]
    location = [i["location"] for i in batch]
    return dict(x=x, y=y, files=files, coords=coords, location=location)


class S2Dataset(torch.utils.data.Dataset):
    """
    Base S2 Dataset Class
    """

    def __init__(
        self,
        annotations: Dict[str, Dict],
        chip_shape: Tuple[int, int],
        seq_len: int,
        batch_size: int,
        batches_per_epoch: int,
        transforms: T.Compose,
    ):
        self.annotations = annotations
        self.chip_shape = chip_shape
        self.seq_len = seq_len
        self.transforms = transforms
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.sites = list(annotations.keys())
        self.idx2site = dict(enumerate(self.sites))
        self.site2idx = {s: i for i, s in enumerate(self.idx2site)}
        self.indices = range(len(self.sites))
        self.pos_indices = [i for i, site in enumerate(self.sites) if self.annotations[site]["bbox"] is not None]

    def load_chip(self, site: str, image: str, coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        """Load a chip given pixel coordinates and image"""
        raise NotImplementedError

    def get_transform(self, site: str, image: str) -> np.ndarray:
        """Return the geotransform array for conversion of pixel <-> lat/lon"""
        raise NotImplementedError

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        """Return the shape of image"""
        raise NotImplementedError

    def get_negative_sample(self, site: str, images: List[str]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get random coordinates for a negative stack of images"""
        shape = self.get_image_shape(site, images[0])
        coords = get_random_coords(image_shape=shape, chip_shape=self.chip_shape[1:])
        return [coords] * len(images)

    def get_positive_sample(
        self, site: str, images: List[str], bbox: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Iterate through sequence of images and
        get pixel coordinates given a lat/lon bbox
        """
        coords = []
        for image in images:
            shape = self.get_image_shape(site, image)
            transform = self.get_transform(site, image)
            c = get_latlon_coords(image_shape=shape, bbox=bbox, transform=transform, target_shape=self.chip_shape[1:])
            coords.append(c)

        return coords

    def __len__(self) -> int:
        return self.batch_size * self.batches_per_epoch

    def __getitem__(self, label: int) -> Dict:

        if label == 0:
            idx = random.sample(self.indices, k=1)[0]
        else:
            idx = random.sample(self.pos_indices, k=1)[0]

        site = self.idx2site[idx]
        images = self.files[idx]

        # Sample a random sequence of frames
        # from within entire timeline for site
        t = random.randint(0, len(images) - self.seq_len)
        images = images[t : t + self.seq_len]

        # Negative sample (random)
        if label == 0:
            coords = self.get_negative_sample(site, images)

        # Positive sample (from bbox coordinates)
        else:
            bbox = self.annotations[site]["bbox"]
            coords = self.get_positive_sample(site, images, bbox)

        # Load chips given pixel coords
        chips = [self.load_chip(site, image, coord) for image, coord in zip(images, coords)]

        # Error handle chips which have no shape for transform reasons
        # This seems to correlate with images being full of bad pixels (-9999)
        for i in range(len(chips)):
            if 0 in chips[i].shape:
                chips[i] = np.zeros(shape=self.chip_shape)

        # Preprocess chips and stack into cube
        cubes = [self.transforms(chip) for chip in chips]

        return dict(x=torch.stack(cubes, dim=0), y=torch.tensor(label), files=images, coords=coords, location=site)


class VRTDataset(S2Dataset):
    """
    HLS Dataset loaded from VRT files
    """

    def __init__(self, vrts: List[List[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = vrts

    def get_transform(self, site: str, image: str) -> np.ndarray:
        with rasterio.open(image) as dataset:
            transform = np.array(dataset.transform)
        return transform

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        with rasterio.open(image) as dataset:
            shape = dataset.shape
        return shape

    def load_chip(self, site: str, image: str, coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        return load_vrt(image, coords)


class HDF5Dataset(S2Dataset):
    """
    HLS Dataset loaded from HDF5 preprocessed file
    """

    def __init__(self, hdf5_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf5_path = hdf5_path
        self.files = self.get_files()

    def get_files(self) -> List[List[str]]:
        files = []
        with h5py.File(self.hdf5_path, "r") as f:
            for site in self.sites:
                files.append(list(f["images"][site].keys()))
        return files

    def get_transform(self, site: str, image: str) -> np.ndarray:
        with h5py.File(self.hdf5_path, "r") as f:
            transform = f["transforms"][site][image][:].flatten()
        return transform

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        with h5py.File(self.hdf5_path, "r") as f:
            shape = f["images"][site][image].shape[1:]
        return shape

    def load_chip(self, site: str, image: str, coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = coords
        with h5py.File(self.hdf5_path, "r") as f:
            chip = f["images"][site][image][:, xmin:xmax, ymin:ymax]
        return chip
