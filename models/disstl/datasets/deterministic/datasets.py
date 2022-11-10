import os
import random
from typing import Tuple, List, Dict
import copy
import h5py
from osgeo import gdal
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
from collections import defaultdict

from disstl.datasets.utils import get_latlon_coords, get_random_coords


from abc import ABC, abstractmethod


class IndexedDatasetBase(ABC, torch.utils.data.Dataset):
    def __init__(
        self,
        fields: Dict = {},
        maps: Dict = {},
        tuples: List[
            Tuple[str, List[str], List[Tuple[Tuple[int, int], Tuple[int, int]]]]
        ] = [],  # [site, image_seq, coords]
        sample_indices: List[int] = [],
        metadata_by_index: List[Dict] = [],
        metadata_by_key: Dict = {},
    ):
        self._fields = fields
        self._maps = maps
        self.sample_indices = sample_indices
        self._tuples = tuples
        self._metadata_by_index = metadata_by_index
        self._metadata_by_key = metadata_by_key

    @property
    def sample_indices(self):
        """ """
        try:
            self._sample_indices
        except AttributeError:
            self._sample_indices = list(range(len(self.sample_tuples)))
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, sample_indices: List[int] = []):
        """ """
        self._sample_indices = sample_indices

    @property
    def fields(self):
        return self._fields

    @property
    def maps(self):
        return self._maps

    @property
    def sample_tuples(self):
        """a tuple is: [site, image_seq, coords], used for:"""
        try:
            self._sample_tuples
        except AttributeError:
            self._sample_tuples = [self._tuples[index] for index in self.sample_indices]
        return self._sample_tuples

    @property
    def sample_metadata_by_index(self):
        """ """
        try:
            self._sample_metadata_by_index
        except AttributeError:
            self._sample_metadata_by_index = [self._metadata_by_index[index] for index in self.sample_indices]
        return self._sample_metadata_by_index

    @property
    def sample_metadata_by_key(self):
        """ """
        try:
            self._sample_metadata_by_key
        except AttributeError:
            sample_metadata_by_key = {}
            for key, val in self._metadata_by_key.items():
                sample_metadata_by_key[key] = [val[index] for index in self.sample_indices]
            self._sample_metadata_by_key = sample_metadata_by_key
        return self._sample_metadata_by_key

    @property
    def metadata_distribution_by_id(self, normalized=True):
        """ """
        metadata_distribution_by_id = {}
        for key, val in self.sample_metadata_by_key.items():
            hist = np.histogram(val, np.arange(0, len(self.fields[key]) + 1), density=False)[0]
            assert len(hist) == len(self.fields[key])
            metadata_distribution_by_id[key] = dict(zip(self.fields[key], hist))
        return metadata_distribution_by_id

    @property
    def metadata_distribution_by_label(self, normalized=True):
        """ """
        metadata_distribution_by_label = {}
        for key, val in self.sample_metadata_by_key.items():
            field_label = key.split("_id")[0]
            hist = np.histogram(val, np.arange(0, len(self.fields[key]) + 1), density=False)[0]
            assert len(hist) == len(self.fields[key])
            metadata_distribution_by_label[field_label] = dict(zip(self.fields[field_label], hist))
        return metadata_distribution_by_label

    @abstractmethod
    def __getitem__(self, sample_index):
        """ """

    @abstractmethod
    def __len__(self):
        """ """


class IndexedDataset(IndexedDatasetBase):
    def __init__(
        self,
        annotations: Dict,
        chip_shape: Tuple[int, int],
        seq_len: int,
        transforms: T.Compose,
        n_samples_per_site: int,
        seed: int = 5,
    ):
        self.annotations = annotations
        self.chip_shape = chip_shape
        self.seq_len = seq_len
        self.transforms = transforms

        self.pos_site_ids = [
            i for i, (site, metadata) in enumerate(self.annotations.items()) if metadata["bbox"] is not None
        ]

        fields = {}
        field_keys = ["site", "class", "type", "country"]  # these are what we want to track
        metadata_keys = ["type", "country"]  # these live in the annotations
        # compute fields that require special attention
        sites = sorted(list(self.annotations.keys()))
        fields["site"] = sites
        fields["class"] = ["no-construction", "construction"]

        self.files = self.get_files(sites=sites)

        # automatically populate unpopulated fields and build maps
        maps = {}
        for key in field_keys:
            if key not in fields:
                fields[key] = sorted(list(set([metadata[key] for site, metadata in self.annotations.items()])))
            fields[f"{key}_id"] = list(range(len(fields[key])))
            maps[f"{key}2id"] = dict(zip(fields[key], fields[f"{key}_id"]))
            maps[f"id2{key}"] = dict(zip(fields[f"{key}_id"], fields[key]))

        pre_sampling_state = random.getstate()
        tuples = []
        metadata_by_index = []
        metadata_by_key = defaultdict(list)
        for site, metadata in self.annotations.items():
            site_id = maps["site2id"][site]
            class_id = int(site_id in self.pos_site_ids)
            images = self.files[site_id]
            random.seed(seed)
            for n in range(n_samples_per_site):
                # Sample a random sequence of frames
                # from within entire timeline for site
                t = random.randint(0, len(images) - self.seq_len)
                image_seq = images[t : t + self.seq_len]

                # Negative sample (random)
                if class_id == 0:
                    coords = self.get_negative_sample(site, image_seq)

                # Positive sample (from bbox coordinates)
                else:
                    bbox = self.annotations[site]["bbox"]
                    coords = self.get_positive_sample(site, image_seq, bbox)

                tuples.append((site, image_seq, coords))
                seq_metadata = {f"{key}_id": maps[f"{key}2id"][metadata[key]] for key in metadata_keys}
                seq_metadata.update(dict(site_id=site_id, class_id=class_id))
                metadata_by_index.append(seq_metadata)
                for key, val in seq_metadata.items():
                    metadata_by_key[key].append(val)

        metadata_by_key = dict(metadata_by_key)
        # return random module to it's pre data sampling state
        random.setstate(pre_sampling_state)
        sample_indices = list(range(len(tuples)))
        super().__init__(
            fields=fields,
            maps=maps,
            tuples=tuples,
            sample_indices=sample_indices,
            metadata_by_index=metadata_by_index,
            metadata_by_key=metadata_by_key,
        )

    def load_chip(self, site: str, image: str, coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        """Load a chip given pixel coordinates and image"""
        raise NotImplementedError

    def get_transform(self, site: str, image: str) -> np.ndarray:
        """Return the geotransform array for conversion of pixel <-> lat/lon"""
        raise NotImplementedError

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        """Return the shape of image"""
        raise NotImplementedError

    # TODO: complete this method signature
    def get_files(self, sites: list):
        """Return the list of files"""
        raise NotImplementedError

    def get_negative_sample(self, site: str, images: List[str]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get random coordinates for a negative stack of images"""
        shape = self.get_image_shape(site, images[0])
        coords = get_random_coords(image_shape=shape, chip_shape=self.chip_shape)
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

    def __getitem__(self, sample_idx: int) -> Dict:

        # Load chips given index into the dataset
        site, image_seq, coords = self.sample_tuples[sample_idx]
        chips = [self.load_chip(site, image, coord) for image, coord in zip(image_seq, coords)]

        # Preprocess chips and stack into cube
        cubes = [self.transforms(chip) for chip in chips]

        item = dict(
            sample_idx=sample_idx,
            x=torch.stack(cubes, dim=0),
            y=torch.tensor(self.sample_metadata_by_index[sample_idx]["class_id"]),
            files=image_seq,
            coords=coords,
            **self.sample_metadata_by_index[sample_idx],
        )
        return item

    def __len__(self) -> int:
        return len(self.sample_indices)


class SubsetDataset(IndexedDatasetBase):
    def __init__(self, dataset: IndexedDataset = None, sample_indices: list = []):
        self._dataset = dataset
        super().__init__(
            fields=self._dataset.fields,
            maps=self._dataset.maps,
            tuples=self._dataset._tuples,
            sample_indices=sample_indices,
            metadata_by_index=self._dataset._metadata_by_index,
            metadata_by_key=self._dataset._metadata_by_key,
        )

    def __getitem__(self, sample_index):
        return self._dataset[sample_index]

    def __len__(self):
        return len(self.sample_indices)


class FineTuneDataset(IndexedDatasetBase):
    def __init__(self, dataset: IndexedDataset = None, uncertainty_threshold: float = 0.5, sentinel_value: int = -100):
        self._dataset = dataset
        super().__init__(
            fields=self._dataset.fields,
            maps=self._dataset.maps,
            tuples=self._dataset._tuples,
            sample_indices=self._dataset.sample_indices,
            metadata_by_index=self._dataset._metadata_by_index,
            metadata_by_key=self._dataset._metadata_by_key,
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.sentinel_value = sentinel_value
        self.gt_labels_by_index = self._dataset.sample_metadata_by_key["class_id"]
        self.pl_labels_by_index = [(self.sentinel_value, 1.0)] * len(self._dataset)  # [(), (), ...]
        self.ft_labels_by_index = [self.sentinel_value] * len(
            self._dataset
        )  # [ft_label at sample 0, ft_label at sample 1, ...]
        self._ft_label_swap = defaultdict(
            list
        )  # {sample_index: (changed_from, changed_to), (changed_from, changed_to), ..., ...}

    @property
    def ft_label_swap(self):
        return dict(self._ft_label_swap)

    def update_pl_labels(self, labeled_indices: dict = {}):
        for sample_index, (label, unc) in labeled_indices.items():
            self.pl_labels_by_index[sample_index] = (label, unc)

            if label >= 0 and unc < self.uncertainty_threshold:
                # update the convenience property ft_labels_by_index, and track the ft swaps
                existing_ft_label = self.ft_labels_by_index[sample_index]
                if existing_ft_label != self.sentinel_value and existing_ft_label != label:
                    self._ft_label_swap[sample_index].append((existing_ft_label, label))

                self.ft_labels_by_index[sample_index] = label

    def __getitem__(self, sample_index):
        item = self._dataset[sample_index]
        item.update(
            dict(
                pl_label=torch.tensor(self.pl_labels_by_index[sample_index]),
                ft_label=torch.tensor(self.ft_labels_by_index[sample_index]),
                gt_label=torch.tensor(self.gt_labels_by_index[sample_index]),
            )
        )
        return item

    def __len__(self):
        return len(self._dataset)


class HDF5Dataset(IndexedDataset):
    """
    IndexedDataset loaded from HDF5 preprocessed file
    """

    def __init__(self, hdf5_path: str, *args, **kwargs):

        self.hdf5_path = hdf5_path

        with h5py.File(self.hdf5_path, "r") as f:
            self.f = f
            super().__init__(*args, **kwargs)

    def get_files(self, sites: list = []) -> List[List[str]]:
        files = []
        # with h5py.File(self.hdf5_path, "r") as f:
        for site in sites:
            files.append(list(self.f["images"][site].keys()))
        return files

    def get_transform(self, site: str, image: str) -> np.ndarray:
        # with h5py.File(self.hdf5_path, "r") as f:
        transform = self.f["transforms"][site][image][:].flatten()
        return transform

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        # with h5py.File(self.hdf5_path, "r") as f:
        shape = self.f["images"][site][image].shape[1:]
        return shape

    def load_chip(self, site: str, image: str, coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = coords
        with h5py.File(self.hdf5_path, "r") as f:
            chip = f["images"][site][image][:, xmin:xmax, ymin:ymax]
            # Error handle chips which have no shape or a shape not equal to self.chip_shape for transform reasons
            # This seems to correlate with images being full of bad pixels (-9999)
            # if 0 in chip.shape or chip.shape[1:] != self.chip_shape:
            #    chip = np.zeros([chip.shape[0], *self.chip_shape])
        return chip


class VirginiaDataset(IndexedDatasetBase):
    """
    IndexedDataset loaded from HDF5 preprocessed file
    """

    def __init__(
        self,
        base_path: str = None,
        annotations: dict = None,
        transforms: T.Compose = None,
        min_sequence_length: int = 10,
        max_sequence_length: int = 10,
        sequence_length: int = 10,
        seed: int = 5,
        precomputed: bool = False,
        precomputed_fname: str = None,
    ):

        self.base_path = base_path
        self.transforms = transforms if transforms is not None else T.Compose([])
        self.annotations = annotations if annotations is not None else {}
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.sequence_length = sequence_length
        self.precomputed = precomputed
        self.precomputed_fname = precomputed_fname
        noconstruction_site_base_dir = os.path.join(base_path, "Image_NonConstruction")
        construction_site_base_dir = os.path.join(base_path, "Image_Construction")
        self.class_paths = {"construction": construction_site_base_dir, "no-construction": noconstruction_site_base_dir}
        sites, files = self.get_sites_files()

        max_computed_seq_length = 0
        for seq_files in files.values():
            seq_len = max([len(image_seq) for image_seq in seq_files.values()])
            if max_computed_seq_length < seq_len:
                max_computed_seq_length = seq_len
        self.max_sequence_length = min([max_computed_seq_length, max_sequence_length])
        # read metadata
        fields = {}
        field_keys = [
            "class",
            "site",
            # "year", # this is chip-level information
            # "ordinal_date", # this is chip-level information
        ]  # these are what we want to track
        annotation_keys = [
            "area_to_be_disturbed_acres",
            "est_project_end_date",
            "est_project_start_date",
            "latitude",
            "longitude",
            "location_city",
            "permit_number",
            "total_area_of_development_acres",
        ]  # these live in the annotations
        fields["class"] = sorted(list(files.keys()))

        # compute fields that require special attention
        fields["site"] = sites
        fields["class"] = ["no-construction", "construction"]
        # automatically populate unpopulated fields and build maps
        maps = {}
        for key in field_keys:
            fields[f"{key}_id"] = list(range(len(fields[key])))
            maps[f"{key}2id"] = dict(zip(fields[key], fields[f"{key}_id"]))

        # undo the site to site_id mapping that was automatically created above,
        # no obvious need to re-enumerate the site labels
        fields["site_id"] = [int(site) for site in sites]
        maps["site2id"] = dict(zip(fields["site"], fields["site_id"]))

        if self.precomputed is True:
            ds = torch.load(self.precomputed_fname)
            tuples = ds["tuples"]
            sample_indices = ds["sample_indices"]
            metadata_by_index = ds["metadata_by_index"]
            metadata_by_key = ds["metadata_by_key"]
            self.chips_by_image_fname = ds["chips_by_image_fname"]
        else:
            # generate samples here
            tuples = []
            metadata_by_index = []
            metadata_by_key = defaultdict(list)
            for class_label, sites_dict in files.items():
                for site, site_list in sites_dict.items():
                    tuples.append((site, site_list))
                    site_id = maps["site2id"][site]
                    seq_metadata = {
                        "class_id": maps["class2id"][class_label],
                        "site_id": site_id,
                        **{key: annotations[site_id][key] for key in annotation_keys},
                    }
                    metadata_by_index.append(seq_metadata)
                    for key, val in seq_metadata.items():
                        metadata_by_key[key].append(val)
            metadata_by_key = dict(metadata_by_key)
            sample_indices = list(range(len(tuples)))

        super().__init__(
            fields=fields,
            maps=maps,
            tuples=tuples,
            sample_indices=sample_indices,
            metadata_by_index=metadata_by_index,
            metadata_by_key=metadata_by_key,
        )

    def get_sites_files(self) -> List[List[str]]:
        files = defaultdict(dict)
        sites = set()
        for class_label, class_path in self.class_paths.items():
            sites_from_disk = os.listdir(os.path.join(self.base_path, class_path))
            for site_label in sites_from_disk:
                sites.add(site_label)
                image_fname_sequence = sorted(os.listdir(os.path.join(class_path, site_label)))
                len_image_sequence = len(image_fname_sequence)
                if len_image_sequence < self.min_sequence_length:
                    continue
                elif len_image_sequence > self.max_sequence_length:
                    image_indices = sorted(random.sample(range(len_image_sequence), k=self.max_sequence_length))
                else:
                    image_indices = range(len_image_sequence)
                image_fname_sequence = [image_fname_sequence[i] for i in image_indices]
                files[class_label][site_label] = []
                for tiff_fname in image_fname_sequence:
                    files[class_label][site_label].append(os.path.join(class_path, site_label, tiff_fname))
        files = dict(files)

        sites = sorted([int(site) for site in sites])
        sites = [str(site) for site in sites]  # return to string value since this is a label, not an id,
        # and it will be used to refer to a file path

        return sites, files

    def get_transform(self, site: str, image: str) -> np.ndarray:
        # with h5py.File(self.hdf5_path, "r") as f:
        transform = self.f["transforms"][site][image][:].flatten()
        return transform

    def get_image_shape(self, site: str, image: str) -> Tuple[int, int]:
        # with h5py.File(self.hdf5_path, "r") as f:
        shape = self.f["images"][site][image].shape[1:]
        return shape

    def load_chip(self, image_fname: str = None, with_QA: bool = False) -> np.ndarray:
        if self.precomputed is True:
            chip, QA = self.chips_by_image_fname[image_fname]
            if with_QA is True:
                return chip, QA
            elif with_QA is False:
                return chip
        else:
            gdal_ds = gdal.Open(os.path.join(image_fname), gdal.GA_ReadOnly)
            ds_subsets = gdal_ds.GetSubDatasets()
            chip = torch.zeros((len(ds_subsets) - 1, 40, 40))  # the last band is QA
            for i, (sub_ds_name, _) in enumerate(ds_subsets[:-1]):
                sub_ds = gdal.Open(sub_ds_name, gdal.GA_ReadOnly)
                band = sub_ds.GetRasterBand(1).ReadAsArray()
                chip[i, :, :] = torch.from_numpy(band)
            if with_QA is True:
                QA_ds = gdal.Open(ds_subsets[-1][0], gdal.GA_ReadOnly)
                QA_band = QA_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
                QA = torch.from_numpy(QA_band)
                return chip, QA
            elif with_QA is False:
                return chip

    @staticmethod
    def load_annotations(path: str = None):
        df = pd.read_csv(path, index_col="ID", keep_default_na=False)
        og_cols = df.columns
        new_columns = [col_name.replace(" ", "_").lower() for col_name in og_cols]
        df.columns = new_columns
        date_cols = ["est_project_start_date", "est_project_end_date"]
        for date_col in date_cols:
            df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y")
            df[date_col] = df[date_col].apply(lambda d: f"{d.year}{d.day_of_year}")
        annotations = df.to_dict(orient="index")
        return annotations

    @staticmethod
    def build_chipset(
        base_path: str = None,
        annotations: dict = None,
        transforms: T.Compose = None,
        min_sequence_length: int = 10,
        max_sequence_length: int = 10,
        sequence_length: int = 10,
        seed: int = 5,
        ds_path: str = None,
        chipset_fname: str = None,
    ):
        ds = VirginiaDataset(
            base_path=base_path,
            annotations=annotations,
            transforms=transforms,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            sequence_length=sequence_length,
            seed=seed,
            precomputed=False,
            precomputed_fname=None,
        )
        ds.save(ds_path=ds_path, chipset_fname=chipset_fname)
        return None

    def save(self, ds_path: str = None, chipset_fname: str = "virginia_ds.torch"):
        ds = {}
        ds["tuples"] = self._tuples
        ds["sample_indices"] = self.sample_indices
        ds["metadata_by_index"] = self._metadata_by_index
        ds["metadata_by_key"] = self._metadata_by_key
        ds["chips_by_image_fname"] = {}
        for sample_idx in tqdm(self.sample_indices):
            _, image_seq = self.sample_tuples[sample_idx]
            for i, image_fname in enumerate(image_seq):
                chip, QA = self.load_chip(image_fname, with_QA=True)
                ds["chips_by_image_fname"][image_fname] = (chip, QA)
        os.makedirs(ds_path, exist_ok=True)
        torch.save(ds, os.path.join(ds_path, chipset_fname))
        print(f"chipset saved to {ds_path}/{chipset_fname}")

    def __getitem__(self, sample_idx: int):
        # Load chips given index into the dataset
        _, image_seq = self.sample_tuples[sample_idx]
        # chips = [self.load_chip(image_fname) for image_fname in image_seq]
        chips = []
        QAs = []
        len_image_seq = len(image_seq)
        for i, image_fname in enumerate(image_seq):
            chip, QA = self.load_chip(image_fname, with_QA=True)
            chips.append(chip)
            QAs.append(QA)

        # Preprocess chips and stack into cube
        cubes = [self.transforms(chip) for chip in chips]

        y = self.sample_metadata_by_index[sample_idx]["class_id"]
        files = image_seq
        # pad out all data structures to have length self.max_sequence_length
        length_difference = self.max_sequence_length - len_image_seq
        nan_cube = np.empty(cubes[-1].shape)
        nan_cube.fill(np.nan)
        cubes.extend([torch.from_numpy(nan_cube)] * length_difference)
        QAs.extend([torch.zeros(QAs[-1].shape, dtype=QAs[-1].dtype)] * length_difference)
        files.extend([files[-1]] * length_difference)

        item = dict(
            sample_idx=sample_idx,
            x=torch.stack(cubes, dim=0),
            y=y,
            files=files,
            QA=QAs,
            # coords=coords,
            # **self.sample_metadata_by_index[sample_idx],
        )
        return item

    def __len__(self):
        return len(self.sample_indices)


def split_indexed_dataset(dataset: IndexedDataset = None, fraction: float = 1.0, shuffle=True, seed: int = 5):
    indices = copy.deepcopy(dataset.sample_indices)
    if shuffle is True:
        pre_shuffle_state = random.getstate()
        random.seed(seed)
        random.shuffle(indices)
        random.setstate(pre_shuffle_state)
    split_index = np.floor(fraction * len(indices)).astype(int)
    return (
        SubsetDataset(dataset, sample_indices=indices[:split_index]),
        SubsetDataset(dataset, sample_indices=indices[split_index:]),
    )
