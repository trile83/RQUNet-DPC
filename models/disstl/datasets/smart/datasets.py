import json
import hashlib
import os
import pickle
import random
import traceback

from bidict import bidict
from dataclasses import dataclass
from functools import cached_property
from math import comb
from typing import List, Optional, Tuple, Union
from datetime import timedelta, datetime
from functools import partial
import multiprocessing
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import h5py
import numpy as np
import torch
import torchvision.transforms as T

from disstl.datasets.transforms import Pad, ToTensor
from disstl.standards import STD_DATE_FORMAT
from disstl.datasets.utils import pixel_to_latlon, get_date_difference, days_to_ymw
from disstl.annotations.list_of_cleared_regions import CLEARED_REGIONS


@dataclass
class ActivityChunk:
    region_id: str
    region_idx: int
    window: tuple
    coords: Tuple[List[float], List[float]]
    image_seq: np.ndarray
    seq_index: int
    site_seq: np.ndarray = None
    site: int = None
    activity_id_seq: np.ndarray = None
    activity_bool: np.ndarray = None
    next_phase_id: int = None
    next_phase_date: list = None


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return np.array([pool[i] for i in indices])


def collate_fn(batch):
    x = torch.stack([i["x"] for i in batch], dim=0)
    image_dates = [i["image_dates"] for i in batch]
    windows = [i["windows"] for i in batch]
    coords = [i["coords"] for i in batch]
    return dict(x=x, image_dates=image_dates, windows=windows, coords=coords)


class MTMSDatasetBase(torch.utils.data.Dataset):
    @cached_property
    def windows(self):
        """Extract coordinates using sliding window generator"""
        _, h, w = self.cube_shape
        stride_h, stride_w = self.stride
        chip_h, chip_w = self.chip_shape

        windows = []
        for i in range(0, h, stride_h):
            for j in range(0, w, stride_w):
                window = ((i, min(h, i + chip_h)), (j, min(w, j + chip_w)))
                windows.append(window)

        return np.array(windows)

    @cached_property
    def coords(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return [pixel_to_latlon(w, self.cube_transform) for w in self.windows]

    @cached_property
    def summary(self):
        raise NotImplementedError

    @cached_property
    def activity_chunks(self):
        # perhaps there are cached activity chunks
        if self.use_cache_dataset_preprocessing and os.path.isfile(self.cache_filepath):
            cached = self.read_dataset_preprocessing()
            activity_chunks = cached["activity_chunks"]
        else:
            # nothing cached, compute activity chunks and cache them
            activity_chunks = self._compute_activity_chunks()
            if self.cache_filepath is not None:
                self.cache_dataset_preprocessing(preprocessed_data=dict(activity_chunks=activity_chunks))
        return activity_chunks

    def _compute_activity_chunks(self) -> List[ActivityChunk]:
        raise NotImplementedError

    @staticmethod
    def set_cache_filepath(cache_filepath):
        return f"{cache_filepath}.pkl"

    def cache_dataset_preprocessing(self, preprocessed_data: dict = None):
        with open(self.cache_filepath, "wb") as f:
            pickle.dump(preprocessed_data, f)

    def read_dataset_preprocessing(self) -> dict:
        with open(self.cache_filepath, "rb") as f:
            cached = pickle.load(f)
        return cached

    @cached_property
    def spectral_index_transforms(self) -> List[torch.nn.Module]:
        """Instantiate a list of spectral transform objects from disstl.datasets.indices based on what was asked for
        and where the bands are in this cube. The band indices live in self.band_name2idx
        """

        # TODO: This functionality should be reverted back to an Append transform rather than an instance method.
        #  Perhaps spectral index functions could take the band_name2idx dict and the full image array
        #  instead of individual bands. This may allow simpler instantiation of those classes.
        #  See also: self.compute_spectral_indices()
        spectral_index_transforms = []
        if self.spectral_indices is not None:
            for index_transform in self.spectral_indices:
                band_args = [self.band_name2idx[band] for band in index_transform.ordered_bands]
                spectral_index_transforms.append(index_transform(*band_args))
        # else spectral_index_transforms is fine as an empty list
        return spectral_index_transforms

    def compute_spectral_indices(self, x) -> torch.Tensor:
        """
        Compute a torch tensor of spectral indices computed from x
        """
        # TODO: This functionality should be reverted back to an Append transform rather than an instance method.
        #  See also the property: self.spectral_index_transforms
        assert len(self.spectral_index_transforms) > 0, "self.spectral_index_transforms is emtpy, nothing to compute"
        return torch.stack([spectral_transform(x) for spectral_transform in self.spectral_index_transforms], dim=1)

    def bitwise_or(self, mask: torch.Tensor, values: Union[List[int], List[float]]) -> torch.Tensor:
        """chained bitwise_or:
            self.bitwise_or(mask, [1,2,3]) performs bitwise_or(bitwise_or(mask==1, mask==2), mask==3)

        Note: Accommodates values of length 1 by returning mask == values[0]
        """
        m = mask == values[0]
        if len(values) > 1:
            for val in values[1:]:
                m = torch.bitwise_or(m, mask == val)
        return m

    @property
    def preallocated_stack(self):
        raise NotImplementedError

    @staticmethod
    def compute_seq_start_inds(start_idx, end_idx, sequence_draws, seq_len):
        try:
            if start_idx >= end_idx:
                seq_start_inds = np.array([start_idx])
            else:
                seq_start_inds = np.random.choice(
                    range(start_idx, end_idx), size=min(sequence_draws, end_idx - start_idx), replace=False
                )
        except BaseException:
            seq_start_inds = np.array([start_idx])

            error = traceback.format_exc()
            print(
                f"Exception in compute_seq_start_inds: {error}\n"
                f"start_idx: {start_idx}\n"
                f"end_idx: {end_idx}\n"
                f"sequence_draws: {sequence_draws}\n"
                f"seq_len: {seq_len}"
            )

        return seq_start_inds

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.activity_chunks)

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.name


class AOIDataset(MTMSDatasetBase):
    def __init__(
        self,
        chip_shape: Tuple[int, int] = None,
        stride: Optional[Tuple[int, int]] = None,
        band_names: List[str] = None,
        seq_len: int = 15,
        temporal_sampling_in_weeks: int = 2,
        fraction_of_windows_to_use: float = 1.0,
        data_seed: int = 42,
        transforms: T.Compose = None,
        spectral_indices: List = [],
        file_handle: h5py.File = None,
        metadata: dict = None,
        cache_filepath: str = None,
        region_id2idx: dict = None,
        use_cache_dataset_preprocessing: bool = True,
        n_sequence_draws: int = 3,
        **kwargs,
    ):
        self.file_handle = file_handle
        self.metadata = metadata
        self.cache_filepath = super().set_cache_filepath(cache_filepath)
        self.use_cache_dataset_preprocessing = use_cache_dataset_preprocessing

        # Location (on earth) of cube
        self.region_id = self.metadata["cube"]["region_id"]
        self.region_id2idx = region_id2idx
        self.region_idx = region_id2idx[self.region_id]
        print(f"\n-----region {self.region_idx}: {self.region_id} --------\n")

        # Sampling parameters
        self.transforms = transforms
        self.spectral_indices = spectral_indices if spectral_indices else []
        self.chip_shape = chip_shape
        self.stride = chip_shape if stride is None else stride
        # get map from requested band_names to their indices in the cube
        self.band_name2idx = bidict(
            {
                band_name: self.metadata["cube"]["bands"][band_name]
                for band_name in band_names
                if band_name in self.metadata["cube"]["bands"]
            }
        )
        self.band_names = list(self.band_name2idx.keys())
        self.band_indices = list(self.band_name2idx.values())
        self.num_bands = len(self.band_names)
        self.num_indices = len(self.spectral_index_transforms)
        self.num_channels = self.num_bands + self.num_indices

        # Figure out how many image dates to use
        self.image_count = self.metadata["cube"]["count"]
        if isinstance(seq_len, str) and seq_len.lower() == "all":
            self.seq_len = self.image_count
        elif isinstance(seq_len, int) and seq_len > self.image_count:
            self.seq_len = self.image_count
        else:
            self.seq_len = seq_len

        self.temporal_sampling_in_weeks = temporal_sampling_in_weeks
        assert (
            0.01 <= fraction_of_windows_to_use <= 1.0
        ), "fraction_of_windows_to_use must be between 0.01 (1%) and 1.0 (100%)"
        self.n_sequence_draws = n_sequence_draws
        self.fraction_of_windows_to_use = fraction_of_windows_to_use

        self.scl_mask_value_map = bidict({int(k): v for k, v in self.metadata["cube"]["scl_mask_value_map"].items()})
        self.scl_cloud_values = [
            self.scl_mask_value_map.inverse["CLOUD_MEDIUM_PROBABILITY"],
            self.scl_mask_value_map.inverse["CLOUD_HIGH_PROBABILITY"],
        ]

        self.cube_transform = self.metadata["cube"]["transform"]
        self.image_dates = np.array(self.file_handle["imagery"])

        # # TODO: make weeks_from_start a cached_property
        # #  and then make use of this computation for temporal sampling (itertools.groupby() may be interesting here)
        # image_datetimes = [datetime.strptime(date, "%Y-%m-%d") for date in self.image_dates]
        # start_date = image_datetimes[0]
        # weeks_from_start = [(date - start_date).days // 7 for date in image_datetimes]

        self.cube_shape = self.metadata["cube"]["shape"]
        self.rng = np.random.default_rng(data_seed)

        # instantiate these default transforms, see self.default_transforms() for their use
        self.pad = Pad(chip_shape)
        self.to_tensor = ToTensor()

    @cached_property
    def summary(self):
        summary = dict(
            region_id=self.region_id,
            region_idx=self.region_idx,
            num_channels=self.num_channels,
            cube_version=self.metadata["version"],
            region_version=self.metadata["cube"]["region_version"],
            time_slice_period=self.metadata["cube"]["time_slice_period"],
            time_slice_count=self.metadata["cube"]["time_slice_count"],
            ci_commit_sha=self.metadata["execution_context"]["ci_commit_sha"],
            timestamp=self.metadata["execution_context"]["timestamp"],
            class_name=self.__class__.__name__,
            num_samples=len(self),
            seq_len=self.seq_len,
        )
        return summary

    def _compute_activity_chunks(self) -> List[ActivityChunk]:
        activity_chunks = []
        for i, window in enumerate(self.windows):
            earliest_start_idx = 0
            latest_start_idx = self.image_count - self.seq_len
            seq_start_inds = self.compute_seq_start_inds(
                start_idx=earliest_start_idx,
                end_idx=latest_start_idx,
                sequence_draws=self.n_sequence_draws,
                seq_len=self.seq_len,
            )
            for start_idx in seq_start_inds:
                seq_inds = np.arange(start_idx, start_idx + self.seq_len)
                activity_chunk = ActivityChunk(
                    region_id=self.region_id,
                    region_idx=self.region_idx,
                    window=window,
                    coords=self.coords[i],
                    image_seq=self.image_dates[seq_inds],
                    seq_index=1,
                )
                activity_chunks.append(activity_chunk)
        return activity_chunks

    def default_transforms(self, x: np.ndarray = None, pad_value=0):
        x = self.pad(x, pad_value=pad_value)
        x = self.to_tensor(x)
        return x

    @property
    def preallocated_stack(self):
        preallocated_stack = dict(
            x=torch.empty(
                (self.seq_len, self.num_channels, self.chip_shape[0], self.chip_shape[1]), dtype=torch.float32
            ),
            cloud_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
        )
        return preallocated_stack

    def load_chip_from_activity_chunks(self, activity_chunk, temporal_stack):
        (xmin, xmax), (ymin, ymax) = activity_chunk.window
        for t, image in enumerate(activity_chunk.image_seq):
            x_t = self.file_handle["imagery"][image]["image"][self.band_indices, xmin:xmax, ymin:ymax]
            temporal_stack["x"][t][: self.num_bands] = self.default_transforms(x_t, pad_value=0)
            # clouds
            scl_mask = self.default_transforms(
                self.file_handle["imagery"][image]["scl_mask"][xmin:xmax, ymin:ymax], pad_value=0
            )
            cloud_mask = self.bitwise_or(scl_mask, self.scl_cloud_values)
            temporal_stack["cloud_mask"][t] = cloud_mask

        return temporal_stack

    def __getitem__(self, idx: int):
        activity_chunk = self.activity_chunks[idx]
        item = dict(
            image_dates=",".join(activity_chunk.image_seq),  # this is awful, list of strings is mangled by dataloader
            region_id=activity_chunk.region_id,
            region_idx=activity_chunk.region_idx,
            window=activity_chunk.window,
            coords=activity_chunk.coords,
            seq_index=activity_chunk.seq_index,
        )
        temporal_stack = self.load_chip_from_activity_chunks(activity_chunk, self.preallocated_stack)
        item.update(temporal_stack)

        item["x"][:, : self.num_bands] = self.transforms(item["x"][:, : self.num_bands])

        if len(self.spectral_indices) > 0:
            item["x"][:, self.num_bands :] = self.compute_spectral_indices(temporal_stack["x"][:, : self.num_bands])

        return item


class AnnotatedDataset(AOIDataset):
    """
    Dataset loaded from HDF5 preprocessed file
    """

    def __init__(
        self,
        chip_shape: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None,
        band_names: List[str] = None,
        seq_len: int = 15,
        transforms: T.Compose = None,
        spectral_indices: List = [],
        file_handle: h5py.File = None,
        metadata: dict = None,
        cache_filepath: str = None,
        use_cache_dataset_preprocessing: bool = True,
        balance_windows: bool = False,
        active_to_inactive_ratio: float = 1.0,  # ratio of active windows to inactive windows for training
        n_active_sequence_draws: int = 1,
        n_inactive_sequence_draws: int = 1,
        region_id2idx: int = None,
        data_seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            chip_shape=chip_shape,
            stride=stride,
            band_names=band_names,
            seq_len=seq_len,
            transforms=transforms,
            spectral_indices=spectral_indices,
            file_handle=file_handle,
            metadata=metadata,
            cache_filepath=cache_filepath,
            region_id2idx=region_id2idx,
            use_cache_dataset_preprocessing=use_cache_dataset_preprocessing,
            data_seed=data_seed,
        )
        # collect annotation sampling parameters
        self.active_to_inactive_ratio = active_to_inactive_ratio
        self.n_active_sequence_draws = n_active_sequence_draws
        self.n_inactive_sequence_draws = n_inactive_sequence_draws

        self.activity_mask_value_map = bidict(
            {int(k): v for k, v in self.metadata["cube"]["activity_mask_value_map"].items()}
        )
        unlabeled_string = "Unlabeled"
        self.activity_UNLABELED_value = self.activity_mask_value_map.inverse[unlabeled_string]
        self.activity_NO_ACTIVITY_value = self.activity_mask_value_map.inverse["No Activity"]
        self.activity_UNKNOWN_value = self.activity_mask_value_map.inverse["Unknown"]
        self.site_mask_value_map = bidict({int(k): v for k, v in self.metadata["cube"]["site_mask_value_map"].items()})
        self.site_UNLABELED_value = self.site_mask_value_map.inverse["UNLABELED"]
        self.is_cleared_region = self.region_id in CLEARED_REGIONS

    @cached_property
    def summary(self):
        summary = super(AnnotatedDataset, self).summary
        # this is a placeholder method,
        # update summary to add anything interesting about annotated datasets here
        return summary

    def _compute_activity_chunks(self) -> List[ActivityChunk]:

        w_site_seqs, w_activity_id_seqs, w_activity_bool, w_unknown_bool = self._get_window_annotation_seqs()

        active_chunks = []
        inactive_chunks = []

        for window_index in np.where(w_activity_bool == 1)[0]:

            # isolate temporal indices where activity is occurring
            active_inds = np.where(w_activity_id_seqs[window_index] != self.activity_UNLABELED_value)[0]
            active_start_idx = active_inds[0]
            active_end_idx = active_inds[-1]

            # get the site in this active window
            site = w_site_seqs[window_index][active_start_idx]

            # draw n_active_sequence_draws in an interval around active_inds
            earliest_start_idx = max(active_start_idx - (self.seq_len - 1), 0)
            if self.image_count == self.seq_len:
                latest_start_idx = earliest_start_idx
            elif active_end_idx == active_start_idx:
                latest_start_idx = earliest_start_idx
            else:
                latest_start_idx = min(active_end_idx - 2, self.image_count - self.seq_len)
            seq_start_inds = self.compute_seq_start_inds(
                start_idx=earliest_start_idx,
                end_idx=latest_start_idx,
                sequence_draws=self.n_active_sequence_draws,
                seq_len=self.seq_len,
            )
            for n, start_idx in enumerate(seq_start_inds):
                active_seq_inds = np.arange(start_idx, start_idx + self.seq_len)
                active_chunk = ActivityChunk(
                    region_id=self.region_id,
                    region_idx=self.region_idx,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=w_site_seqs[window_index][active_seq_inds],
                    site=site,
                    activity_id_seq=w_activity_id_seqs[window_index][active_seq_inds],
                    activity_bool=1,
                    image_seq=self.image_dates[active_seq_inds],
                    seq_index=n,
                )
                active_chunks.append(active_chunk)

            # This completes active chunks for this window.
            # Now draw inactive sequences from this window prior to active_start_idx and after active_end_idx
            if active_start_idx > self.seq_len:  # then we can draw sequences from before activity started
                earliest_start_idx = 0
                latest_start_idx = active_start_idx - self.seq_len
                seq_start_inds = self.compute_seq_start_inds(
                    start_idx=earliest_start_idx,
                    end_idx=latest_start_idx,
                    sequence_draws=self.n_inactive_sequence_draws,
                    seq_len=self.seq_len,
                )
                for n, start_idx in enumerate(seq_start_inds):
                    inactive_seq_inds = np.arange(start_idx, start_idx + self.seq_len)
                    inactive_chunk = ActivityChunk(
                        region_id=self.region_id,
                        region_idx=self.region_idx,
                        window=self.windows[window_index],
                        coords=self.coords[window_index],
                        site_seq=w_site_seqs[window_index][inactive_seq_inds],
                        site=site,
                        activity_id_seq=w_activity_id_seqs[window_index][inactive_seq_inds],
                        activity_bool=0,
                        image_seq=self.image_dates[inactive_seq_inds],
                        seq_index=n,
                    )
                    inactive_chunks.append(inactive_chunk)

            # Now draw inactive sequences from this window after the active_end_idx
            if (
                self.image_count - (active_end_idx + 1)
            ) > self.seq_len:  # then we can draw sequences from after activity started
                earliest_start_idx = active_end_idx + 1
                latest_start_idx = self.image_count - self.seq_len
                seq_start_inds = self.compute_seq_start_inds(
                    start_idx=earliest_start_idx,
                    end_idx=latest_start_idx,
                    sequence_draws=self.n_inactive_sequence_draws,
                    seq_len=self.seq_len,
                )
                for n, start_idx in enumerate(seq_start_inds):
                    inactive_seq_inds = np.arange(start_idx, start_idx + self.seq_len)
                    inactive_chunk = ActivityChunk(
                        region_id=self.region_id,
                        region_idx=self.region_idx,
                        window=self.windows[window_index],
                        coords=self.coords[window_index],
                        site_seq=w_site_seqs[window_index][inactive_seq_inds],
                        site=site,
                        activity_id_seq=w_activity_id_seqs[window_index][inactive_seq_inds],
                        activity_bool=0,
                        image_seq=self.image_dates[inactive_seq_inds],
                        seq_index=n,
                    )
                    inactive_chunks.append(inactive_chunk)

        # Done with all active windows, move to inactive windows
        ######################
        # All active_chunks have been computed, and several inactive chunks may have been computed as well
        # Now compute more inactive chunks based on active_to_inactive_ratio

        inactive_windows = np.where(w_activity_bool == 0)[0]
        if not self.is_cleared_region:
            n_inactive_windows_to_use = 0
        elif self.active_to_inactive_ratio == "unspecified":
            n_inactive_windows_to_use = len(inactive_windows)
        else:
            n_active_chunks = len(active_chunks)
            n_inactive_chunks = max(
                [1, len(inactive_chunks)]
            )  # inactive_chunks may be 0, not good as a denominator below
            B = n_active_chunks / (n_inactive_chunks * self.active_to_inactive_ratio)
            n_inactive_chunks_to_compute = int(B * n_inactive_chunks) - n_inactive_chunks if B > 1 else 0

            n_inactive_windows_to_use = int(n_inactive_chunks_to_compute / self.n_inactive_sequence_draws)

            np.random.shuffle(inactive_windows)

        for window_index in inactive_windows[:n_inactive_windows_to_use]:
            site = w_site_seqs[window_index][0]
            earliest_start_idx = 0
            latest_start_idx = self.image_count - self.seq_len
            seq_start_inds = self.compute_seq_start_inds(
                start_idx=earliest_start_idx,
                end_idx=latest_start_idx,
                sequence_draws=self.n_inactive_sequence_draws,
                seq_len=self.seq_len,
            )
            for n, start_idx in enumerate(seq_start_inds):
                inactive_seq_inds = np.arange(start_idx, start_idx + self.seq_len)
                inactive_chunk = ActivityChunk(
                    region_id=self.region_id,
                    region_idx=self.region_idx,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=w_site_seqs[window_index][inactive_seq_inds],
                    site=site,
                    activity_id_seq=w_activity_id_seqs[window_index][inactive_seq_inds],
                    activity_bool=0,
                    image_seq=self.image_dates[inactive_seq_inds],
                    seq_index=n,
                )
                inactive_chunks.append(inactive_chunk)

        return [*active_chunks, *inactive_chunks]

    def _activity_window_helper(self, idx_window, site_masks, activity_masks):
        # Iterate over the windows that we want to look at
        i, window = idx_window
        (xmin, xmax), (ymin, ymax) = window
        site_seq, activity_id_seq = [], []
        for site_mask, activity_mask in zip(site_masks, activity_masks):
            # Iterate over the masks for all images
            site_seq.append(self.get_label(site_mask[xmin:xmax, ymin:ymax], sentinel_value=self.site_UNLABELED_value))
            activity_submask = activity_mask[xmin:xmax, ymin:ymax]

            activity_id_seq.append(self.get_label(activity_submask, sentinel_value=self.activity_UNLABELED_value))
        site_seq, activity_id_seq = np.array(site_seq), np.array(activity_id_seq)
        activity_bool = int(any(activity_id_seq != self.activity_UNLABELED_value))
        unknown_bool = int(all(activity_id_seq == self.activity_UNKNOWN_value))
        return {
            i: {
                "site_seq": site_seq,
                "activity_id_seq": activity_id_seq,
                "activity_bool": activity_bool,
                "unknown_bool": unknown_bool,
            }
        }

    def _get_window_annotation_seqs(self):
        activity_mask_dtype = self.file_handle["imagery"][self.image_dates[0]]["activity_mask"].dtype
        site_mask_dtype = self.file_handle["imagery"][self.image_dates[0]]["site_mask"].dtype
        activity_masks = np.empty((self.image_dates.size, *self.cube_shape[1:]), dtype=activity_mask_dtype)
        site_masks = np.empty((self.image_dates.size, *self.cube_shape[1:]), dtype=site_mask_dtype)
        for i, image in enumerate(self.image_dates):
            try:
                activity_masks[i, :, :] = self.file_handle["imagery"][image]["activity_mask"]
                site_masks[i, :, :] = self.file_handle["imagery"][image]["site_mask"]
            except KeyError:
                # images in cube extend to dates beyond what has been annotated
                activity_masks = activity_masks[:i]
                site_masks = site_masks[:i]
                self.image_dates = self.image_dates[:i]
                break

        # collapse No Activty to Unlabeled
        activity_masks = np.where(
            activity_masks == self.activity_NO_ACTIVITY_value, self.activity_UNLABELED_value, activity_masks
        )

        site_seq = np.empty((len(self.windows), self.image_dates.size), dtype=site_mask_dtype)
        activity_id_seq = np.empty((len(self.windows), self.image_dates.size), dtype=activity_mask_dtype)
        activity_bool = np.empty((len(self.windows)), dtype=object)
        unknown_bool = np.empty((len(self.windows)), dtype=object)

        with ThreadPool(multiprocessing.cpu_count()) as p:
            results = list(
                tqdm(
                    p.imap(
                        partial(self._activity_window_helper, site_masks=site_masks, activity_masks=activity_masks),
                        enumerate(self.windows),
                    ),
                    desc="Identifying activity chunks per window",
                    total=len(self.windows),
                )
            )

        merged_results = {}
        # merge all the dictionaries
        [merged_results.update(result) for result in results]
        for i in range(len(self.windows)):
            site_seq[i] = merged_results[i]["site_seq"]
            activity_id_seq[i] = merged_results[i]["activity_id_seq"]
            activity_bool[i] = merged_results[i]["activity_bool"]
            unknown_bool[i] = merged_results[i]["unknown_bool"]

        return site_seq, activity_id_seq, activity_bool, unknown_bool

    def get_image_subset(self) -> List[str]:
        sorted_random_subset = sorted(random.sample(self.image_dates, k=min([self.seq_len, self.image_dates.size])))
        return sorted_random_subset

    def get_image_subsets(self):
        return [self.get_image_subset() for w in self.windows]

    def get_label(self, mask, sentinel_value: int = []):
        labels, counts = np.unique(mask, return_counts=True)
        # check if no activity is in labels and if counts of no activity are more than 99% of the counts in the mask
        if len(labels) == 1:
            label = int(labels[0])
        else:
            # return most frequent label that is not the sentinel_value
            most_freq = int(labels[0])
            label = most_freq if most_freq != sentinel_value else int(labels[1])

        return label

    @property
    def preallocated_stack(self):
        preallocated_stack = dict(
            x=torch.empty(
                (self.seq_len, self.num_channels, self.chip_shape[0], self.chip_shape[1]), dtype=torch.float32
            ),
            cloud_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
            site_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
            activity_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
            binary_activity_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
            invalid_data_mask=torch.empty((self.seq_len, self.chip_shape[0], self.chip_shape[1]), dtype=torch.long),
            activity_id=torch.empty(self.seq_len, dtype=torch.long),
            site=torch.empty(self.seq_len, dtype=torch.long),
        )
        return preallocated_stack

    def load_chip_from_activity_chunks(self, activity_chunk, temporal_stack):
        (xmin, xmax), (ymin, ymax) = activity_chunk.window
        for t, image in enumerate(activity_chunk.image_seq):
            x_t = self.file_handle["imagery"][image]["image"][self.band_indices, xmin:xmax, ymin:ymax]
            temporal_stack["x"][t][: self.num_bands] = self.default_transforms(x_t, pad_value=0)

            # invalid_data_mask 0: valid data, 1: invalid_data
            try:
                raw_valid_data_mask = self.file_handle["imagery"][image]["valid_data_mask"][xmin:xmax, ymin:ymax]
            except KeyError:
                raw_valid_data_mask = (x_t.sum(axis=0) > 0).astype(np.int64)

            temporal_stack["invalid_data_mask"][t] = self.default_transforms(1 - raw_valid_data_mask, pad_value=0)

            # clouds
            scl_mask = self.default_transforms(
                self.file_handle["imagery"][image]["scl_mask"][xmin:xmax, ymin:ymax], pad_value=0
            )
            cloud_mask = self.bitwise_or(scl_mask, self.scl_cloud_values)
            temporal_stack["cloud_mask"][t] = cloud_mask

            # sites
            temporal_stack["site_mask"][t] = self.default_transforms(
                self.file_handle["imagery"][image]["site_mask"][xmin:xmax, ymin:ymax],
                pad_value=self.site_UNLABELED_value,
            )

            # activity
            activity_mask_t = self.file_handle["imagery"][image]["activity_mask"][xmin:xmax, ymin:ymax]

            # collapse No Activty to Unlabeled
            activity_mask_t = np.where(
                activity_mask_t == self.activity_NO_ACTIVITY_value, self.activity_UNLABELED_value, activity_mask_t
            )
            temporal_stack["activity_mask"][t] = self.default_transforms(
                activity_mask_t, pad_value=self.activity_UNLABELED_value
            )

            # binary activity mask
            binarized_activity_mask = self.bitwise_or(temporal_stack["activity_mask"][t], [1, 2, 3, 4]).type(
                temporal_stack["activity_mask"].dtype
            )
            # derived from temporal_stack["activity_mask"], no need to apply default_transforms()
            temporal_stack["binary_activity_mask"][t] = binarized_activity_mask

            # scalar labels
            temporal_stack["site"][t] = activity_chunk.site_seq[t]
            temporal_stack["activity_id"][t] = activity_chunk.activity_id_seq[t]

        return temporal_stack

    def __getitem__(self, idx: int):
        activity_chunk = self.activity_chunks[idx]
        item = dict(
            image_dates=",".join(activity_chunk.image_seq),  # this is awful, list of strings is mangled by dataloader
            region_id=activity_chunk.region_id,
            region_idx=activity_chunk.region_idx,
            window=activity_chunk.window,
            coords=activity_chunk.coords,
            activity_bool=activity_chunk.activity_bool,
            seq_index=activity_chunk.seq_index,
        )
        assert len(activity_chunk.image_seq) == self.preallocated_stack["binary_activity_mask"].shape[0]
        temporal_stack = self.load_chip_from_activity_chunks(activity_chunk, self.preallocated_stack)
        item.update(temporal_stack)

        item["x"][:, : self.num_bands] = self.transforms(item["x"][:, : self.num_bands])

        if len(self.spectral_indices) > 0:
            item["x"][:, self.num_bands :] = self.compute_spectral_indices(temporal_stack["x"][:, : self.num_bands])

        return item


class SegmentationDataset(AnnotatedDataset):
    def __init__(
        self,
        chip_shape: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None,
        band_names: List[str] = None,
        seq_len: int = 15,
        transforms: T.Compose = None,
        spectral_indices: List = [],
        file_handle: h5py.File = None,
        metadata: dict = None,
        cache_filepath: str = None,
        use_cache_dataset_preprocessing: bool = True,
        balance_windows: bool = False,
        active_to_inactive_ratio: float = 1.0,  # ratio of active windows to inactive windows for training
        n_active_sequence_draws: int = 1,
        n_inactive_sequence_draws: int = 1,
        region_id2idx: int = None,
        data_seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            chip_shape=chip_shape,
            stride=stride,
            band_names=band_names,
            seq_len=seq_len,
            transforms=transforms,
            spectral_indices=spectral_indices,
            file_handle=file_handle,
            metadata=metadata,
            cache_filepath=cache_filepath,
            use_cache_dataset_preprocessing=use_cache_dataset_preprocessing,
            active_to_inactive_ratio=active_to_inactive_ratio,  # ratio of active to inactive windows for training
            n_active_sequence_draws=n_active_sequence_draws,
            n_inactive_sequence_draws=n_inactive_sequence_draws,
            region_id2idx=region_id2idx,
            data_seed=data_seed,
        )

    def _compute_activity_chunks(self) -> List[ActivityChunk]:

        w_site_seqs, w_activity_id_seqs, w_activity_bool, w_unknown_bool = self._get_window_annotation_seqs()

        active_chunks = []
        inactive_chunks = []

        for window_index in np.where(w_activity_bool == 1)[0]:

            # isolate temporal indices where activity is occurring
            active_inds = np.where(w_activity_id_seqs[window_index] != self.activity_UNLABELED_value)[0]

            if len(active_inds) < self.seq_len:
                # must not be enough time slices of activity to sample from here... move to the next active window
                continue

            active_start_idx = active_inds[0]
            active_end_idx = active_inds[-1]
            pre_active_inds = np.arange(0, active_start_idx) if active_start_idx > 0 else []
            post_active_inds = (
                np.arange(active_end_idx + 1, self.image_count) if active_end_idx < self.image_count else []
            )

            # get the site in this active window
            site = np.unique(w_site_seqs[window_index])[0]

            for n in range(self.n_active_sequence_draws):

                pre_active_seq_inds = np.random.choice(
                    pre_active_inds, size=min(3, len(pre_active_inds)), replace=False
                )
                post_active_seq_inds = np.random.choice(
                    post_active_inds, size=min(3, len(post_active_inds)), replace=False
                )
                k_active_inds = self.seq_len - len(pre_active_seq_inds) - len(post_active_seq_inds)
                active_seq_inds = np.random.choice(
                    active_inds, size=min(k_active_inds, len(active_inds)), replace=False
                )
                seq_inds = sorted([*pre_active_seq_inds, *active_seq_inds, *post_active_seq_inds])

                active_chunk = ActivityChunk(
                    region_id=self.region_id,
                    region_idx=self.region_idx,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=w_site_seqs[window_index][seq_inds],
                    site=site,
                    activity_id_seq=w_activity_id_seqs[window_index][seq_inds],
                    activity_bool=1,
                    image_seq=self.image_dates[seq_inds],
                    seq_index=n,
                )
                active_chunks.append(active_chunk)

        # Done with all active windows, move to inactive windows
        ######################
        # All active_chunks have been computed, and several inactive chunks may have been computed as well
        # Now compute more inactive chunks based on active_to_inactive_ratio

        inactive_windows = np.where(w_activity_bool == 0)[0]
        if not self.is_cleared_region:
            n_inactive_windows_to_use = 0
        elif self.active_to_inactive_ratio == "unspecified":
            n_inactive_windows_to_use = len(inactive_windows)
        else:
            n_active_chunks = len(active_chunks)
            n_inactive_chunks = max(
                [1, len(inactive_chunks)]
            )  # inactive_chunks may be 0, not good as a denominator below
            B = n_active_chunks / (n_inactive_chunks * self.active_to_inactive_ratio)
            n_inactive_chunks_to_compute = int(B * n_inactive_chunks) - n_inactive_chunks if B > 1 else 0

            n_inactive_windows_to_use = int(n_inactive_chunks_to_compute / self.n_inactive_sequence_draws)

            np.random.shuffle(inactive_windows)

        for window_index in inactive_windows[:n_inactive_windows_to_use]:
            site = w_site_seqs[window_index][0]

            for n in range(self.n_inactive_sequence_draws):
                inactive_seq_inds = sorted(np.random.choice(range(self.image_count), size=self.seq_len, replace=False))
                inactive_chunk = ActivityChunk(
                    region_id=self.region_id,
                    region_idx=self.region_idx,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=w_site_seqs[window_index][inactive_seq_inds],
                    site=site,
                    activity_id_seq=w_activity_id_seqs[window_index][inactive_seq_inds],
                    activity_bool=0,
                    image_seq=self.image_dates[inactive_seq_inds],
                    seq_index=n,
                )
                inactive_chunks.append(inactive_chunk)

        return [*active_chunks, *inactive_chunks]


class AnnotatedActivityPredictionDataset(AnnotatedDataset):
    def __init__(self, num_years: int, **kwds):
        self.num_years = num_years
        self.dataset_name = "AnnotatedActivityPrediction"
        super().__init__(**kwds)

    def __getitem__(self, idx: int):
        activity_chunk = self.activity_chunks[idx]
        item = dict(
            image_dates=",".join(activity_chunk.image_seq),  # this is awful, list of strings is mangled by dataloader
            region_id=activity_chunk.region_id,
            window=activity_chunk.window,
            coords=activity_chunk.coords,
            seq_index=activity_chunk.seq_index,
            next_phase_id=activity_chunk.next_phase_id,
        )
        temporal_stack = self.load_chip_from_activity_chunks(activity_chunk, self.preallocated_stack)
        item.update(temporal_stack)

        # item["x"][:, : self.num_bands] = self.transforms(item["x"][:, : self.num_bands])

        # date differences: deltas within each activity
        dates_diff_dict = get_date_difference(item["image_dates"])
        delta_dates_dict = days_to_ymw(dates_diff_dict["delta_days"], num_years=self.num_years)
        delta_dates_dict.update(dates_diff_dict)

        activity_span = [
            delta_dates_dict["dates"][0].strftime(STD_DATE_FORMAT),
            delta_dates_dict["dates"][-1].strftime(STD_DATE_FORMAT),
        ]

        item["phase"] = item["activity_id"][0].item()

        item["activity_duration"] = dict(
            phase=item["phase"],
            y_date=activity_span,
            y_nextphase=activity_chunk.next_phase_id,
            y_year=delta_dates_dict["years"],
            y_month=delta_dates_dict["months"],
            y_week=delta_dates_dict["weeks"],
            y_deltadays=delta_dates_dict["delta_days"],
        )

        if self.spectral_indices and (len(self.spectral_indices) > 0):
            item["x"][:, self.num_bands :] = self.compute_spectral_indices(temporal_stack["x"][:, : self.num_bands])

        return item

    def _compute_activity_chunks(self) -> List[ActivityChunk]:
        w_site_seqs, w_activity_id_seqs, w_activity_bool = self._get_window_annotation_seqs()

        active_chunks = []
        inactive_chunks = []
        n_dates = self.image_dates.size

        for w, window_index in enumerate(np.where(w_activity_bool == 1)[0]):
            # find changes in activity_labels
            active_temporal_index = np.where(w_activity_id_seqs[window_index] != self.activity_UNLABELED_value)[0]
            active_start_date = active_temporal_index[0]
            active_end_date = active_temporal_index[-1]

            site = w_site_seqs[window_index][active_start_date]

            # grab 5 dates prior and 5 dates after
            pre_active_date = active_start_date - 5
            pre_active_date = pre_active_date if pre_active_date >= 0 else active_start_date
            post_active_date = active_end_date + 5
            post_active_date = post_active_date if post_active_date <= n_dates else active_end_date
            active_date_range = list(range(pre_active_date, post_active_date))

            # use all avialable observations
            active_indices = active_date_range
            activity_labels = w_activity_id_seqs[window_index][active_indices]

            # start sequential activity detection & parsing
            dates_arr = self.image_dates[active_indices]

            # encode: activity_labels sequence as symbol_freq_count =
            # (group_symbol, group_n_sequential_observations, index_start_of_group)
            all_image_seq = []
            all_w_site_seq = []
            all_activity_id_seq = []
            symbol_freq_count = []

            row_symbols = activity_labels[:]
            count = 0
            for idx, r in enumerate(row_symbols):
                if idx == 0:
                    symbol = r
                    count = 1
                    position = idx
                else:
                    if np.abs(int(symbol) - int(r)) > 0:
                        symbol_freq_count.append((symbol, count, position))
                        position = idx
                        count = 1
                        symbol = r
                    else:
                        count += 1
            symbol_freq_count.append((symbol, count, position))

            for c, (symbol, freq, position) in enumerate(symbol_freq_count):
                if freq > 1:
                    temp_image_dates = [dates_arr[i] for i in range(position, position + freq)]
                    all_image_seq.append(np.array(temp_image_dates))
                    subset_active_indices = np.array([active_indices[idx] for idx in range(position, position + freq)])
                    all_w_site_seq.append(w_activity_id_seqs[window_index][subset_active_indices])
                    all_activity_id_seq.append(w_activity_id_seqs[window_index][subset_active_indices])

            for n, (image_seq, site_seq, activity_id_seq) in enumerate(
                zip(all_image_seq, all_w_site_seq, all_activity_id_seq)
            ):
                """Description:
                Check if the number of observations collected < or > seq_len; and
                 ensure that these have len = self.seq_len by duplicating or discarding
                 entries (from available dates). The date start is selected randomly,
                 but the end is the last item (i.e., indexed -1) in the set.
                 Else, let it pass unmodified.
                """
                if len(image_seq) < self.seq_len:  # ADD
                    k = self.seq_len - len(image_seq)
                    to_add_observations = np.array(random.choices(image_seq.tolist(), k=k))  # with repetition.
                    image_seq = np.sort(np.concatenate((image_seq, to_add_observations), axis=0))
                elif len(image_seq) > self.seq_len:
                    if self.seq_len < 2:
                        print(
                            f"[WARNING] the requested {self.seq_len} is "
                            "likely not containing enough observations (less than two) "
                            "to accurately represent the temoporal activity patterns."
                        )
                    image_seq = image_seq.tolist()
                    temp = np.array([image_seq[-1]])
                    k = self.seq_len - 1  # n_needed_observations
                    selected_observations = np.array(random.choices(image_seq[1:-1], k=k))
                    image_seq = np.sort(np.concatenate((temp, selected_observations), axis=0))
                # else: do nothing, let it pass.

                # Look ahead to the future: Next Construction Phase
                m = n + 1  # 1: one-step head into the future
                if m < len(all_activity_id_seq):  # get label and date from the m-th sequence (next).
                    next_phase_id = all_activity_id_seq[m][0]
                    next_phase_date = all_image_seq[m][0]
                else:
                    next_phase_id = activity_id_seq[0]  # current phase/activity label
                    next_phase_date = datetime.strptime(image_seq[-1], STD_DATE_FORMAT).date() + timedelta(
                        days=1
                    )  # current date + 1 day
                    next_phase_date = [next_phase_date.strftime(STD_DATE_FORMAT)]

                site_seq = np.array([site_seq[0]] * len(image_seq))
                activity_id_seq = np.array([activity_id_seq[0]] * len(image_seq))

                active_chunk = ActivityChunk(
                    region_id=self.region_id,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=site_seq,
                    site=site,
                    activity_id_seq=activity_id_seq,
                    activity_bool=1,
                    image_seq=image_seq,
                    seq_index=n,
                    next_phase_id=next_phase_id,
                    next_phase_date=next_phase_date,
                )
                active_chunks.append(active_chunk)

            # try to grab an inactive chunks from before activity began
            if active_start_date > self.seq_len:
                inactive_start_date = 0
                inactive_end_date = active_start_date  # go up to but not including the active start date
                inactive_date_range = list(range(inactive_start_date, inactive_end_date))
                for n in range(min([self.n_inactive_sequence_draws, comb(len(inactive_date_range), self.seq_len)])):
                    inactive_indices = random_combination(inactive_date_range, self.seq_len)
                    inactive_chunk = ActivityChunk(
                        region_id=self.region_id,
                        window=self.windows[window_index],
                        coords=self.coords[window_index],
                        site_seq=w_site_seqs[window_index][inactive_indices],
                        site=site,
                        activity_id_seq=w_activity_id_seqs[window_index][inactive_indices],
                        activity_bool=0,
                        image_seq=self.image_dates[inactive_indices],
                        seq_index=n,
                        next_phase_id=0,  # background
                    )
                    inactive_chunks.append(inactive_chunk)

        ######################
        # All active_chunks have been computed, and several inactive chunks may have been computed as well
        # Now compute more inactive chunks based on active_to_inactive_ratio
        n_active_chunks = len(active_chunks)
        n_inactive_chunks = max([1, len(inactive_chunks)])  # inactive_chunks may be 0, not good as a denominator below
        B = n_active_chunks / (n_inactive_chunks * self.active_to_inactive_ratio)
        n_inactive_chunks_to_compute = int(B * n_inactive_chunks) - n_inactive_chunks if B > 1 else 0

        inactive_start_date = 0
        inactive_end_date = self.image_dates.size
        inactive_date_range = list(range(inactive_start_date, inactive_end_date))
        n_inactive_sequence_draws = min([self.n_inactive_sequence_draws, comb(len(inactive_date_range), self.seq_len)])
        n_inactive_chunks_to_compute = int(n_inactive_chunks_to_compute / n_inactive_sequence_draws)
        for window_index in np.where(w_activity_bool == 0)[0][:n_inactive_chunks_to_compute]:
            site = w_site_seqs[window_index][0]
            for n in range(n_inactive_sequence_draws):
                inactive_indices = random_combination(inactive_date_range, self.seq_len)
                inactive_chunk = ActivityChunk(
                    region_id=self.region_id,
                    window=self.windows[window_index],
                    coords=self.coords[window_index],
                    site_seq=w_site_seqs[window_index][inactive_indices],
                    site=site,
                    activity_id_seq=w_activity_id_seqs[window_index][inactive_indices],
                    activity_bool=0,
                    image_seq=self.image_dates[inactive_indices],
                    seq_index=n,
                    next_phase_id=0,  # background
                )
                inactive_chunks.append(inactive_chunk)

        return [*active_chunks, *inactive_chunks]


def from_cube(
    path: str = None,
    chip_shape: Tuple[int, int] = None,
    stride: Optional[Tuple[int, int]] = None,
    band_names: List[str] = None,
    seq_len: int = 15,
    temporal_sampling_in_weeks: int = 2,
    fraction_of_windows_to_use: float = 1.0,
    transforms: T.Compose = None,
    spectral_indices=None,
    with_annotations: bool = False,
    sampling_kwargs: dict = {},
    use_cache_dataset_preprocessing: bool = True,
    data_seed: int = 42,
    satellite: str = None,
    region_id: int = None,
    region_id2idx: dict = None,
    with_activity_prediction: bool = False,
    segmentation_first_last: bool = False,
    num_years: int = None,
    **kwargs,
):

    file_handle = h5py.File(path, "r")
    metadata = parse_metadata(file_handle)

    metadata["cube"].setdefault("satellite", satellite)
    metadata["cube"].setdefault("region_id", region_id)
    metadata["cube"].setdefault("region_version", "NA")

    # return either Annotated Dataset or just AOI Dataset
    dataset_kwargs = dict(
        chip_shape=chip_shape,
        stride=stride,
        band_names=band_names,
        seq_len=seq_len,
        transforms=transforms,
        spectral_indices=spectral_indices,
        file_handle=file_handle,
        metadata=metadata,
        region_id2idx=region_id2idx,
        use_cache_dataset_preprocessing=use_cache_dataset_preprocessing,
        data_seed=data_seed,
        num_years=num_years,
    )
    if use_cache_dataset_preprocessing is True:
        # build filename for precomputing
        hashable_call = "".join(
            [
                str(v)
                for k, v in {
                    "with_annotations": with_annotations,
                    **dict(sorted(dataset_kwargs.items())),
                    **dict(sorted(sampling_kwargs.items())),
                    **metadata,
                }.items()
            ]
        )
        call_hash = hashlib.md5(hashable_call.encode("utf-8")).hexdigest()
        cache_path = os.path.join(os.path.dirname(path), "caches")
        os.makedirs(cache_path, exist_ok=True)
        cache_filepath = os.path.join(cache_path, f'{metadata["cube"]["region_id"]}_{call_hash}')
        dataset_kwargs["cache_filepath"] = cache_filepath

    if metadata["cube"]["annotated"] is True and with_annotations is True:
        if with_activity_prediction:
            return AnnotatedActivityPredictionDataset(**dataset_kwargs, **sampling_kwargs)
        elif segmentation_first_last:
            return SegmentationDataset(**dataset_kwargs, **sampling_kwargs)
        else:
            return AnnotatedDataset(**dataset_kwargs, **sampling_kwargs)
    else:
        return AOIDataset(**dataset_kwargs, **sampling_kwargs)


def parse_metadata(hdf5_handle: str = None):
    metadata = {}
    for key in hdf5_handle["metadata"].keys():
        if isinstance(hdf5_handle["metadata"][key], h5py.Dataset):
            try:
                # metadata[key] = json.loads(hdf5_handle["metadata"][key][0].decode("utf-8"))
                metadata[key] = json.loads(hdf5_handle["metadata"][key][0])
            except json.JSONDecodeError:
                # metadata[key] = hdf5_handle["metadata"][key][0].decode("utf-8")
                metadata[key] = hdf5_handle["metadata"][key][0]
    return metadata
