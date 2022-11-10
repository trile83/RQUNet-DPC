import random
import rasterio
import numpy as np
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
from math import modf

from torch.utils.data import ConcatDataset
from hydra.utils import instantiate

from disstl.standards import STD_DATE_FORMAT


def clamp_row(row: int, height: int, min_value: int = 0) -> int:
    return max(min_value, min(height, int(row)))


def clamp_col(col: int, width: int, min_value: int = 0) -> int:
    return max(min_value, min(width, int(col)))


def clamp_height(row: int, height: int, max_height: int, min_height: int = 0) -> int:
    return max(clamp_row(height + row, max_height) - row, min_height)


def clamp_width(col: int, width: int, max_width: int, min_width: int = 0) -> int:
    return max(clamp_col(width + col, max_width) - col, min_width)


def latlon_to_pixel(coords: Tuple[float, float], transform: List[float]) -> Tuple[int, int]:
    """Convert lat/lon geocoordinates to pixel coordinates"""
    t = transform[:6]
    t = rasterio.transform.Affine(*t)
    row, col = rasterio.transform.rowcol(t, coords[0], coords[1])
    return row, col


def pixel_to_latlon(coords: Tuple[int, int], transform: List[float]) -> Tuple[float, float]:
    """Convert pixel coordinates to lat/lon geocoordinates"""
    # Affine only takes first 2 rows since last row is always [0, 0, 1]
    t = transform[:6]
    t = rasterio.transform.Affine(*t)
    lat, lon = rasterio.transform.xy(t, coords[0], coords[1])
    return lat, lon


def get_random_coords(
    image_shape: Tuple[int, int], chip_shape: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Select random window coordinates"""
    h, w = image_shape
    x = random.randint(0, h - (chip_shape[0] + 1))
    y = random.randint(0, w - (chip_shape[1] + 1))
    coords = ((x, x + chip_shape[0]), (y, y + chip_shape[1]))
    return coords


def get_latlon_coords(
    image_shape: Tuple[int, int],
    bbox: Tuple[Tuple[float, float], Tuple[float, float]],
    transform: np.ndarray,
    target_shape: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get pixel coordinates from geocoordinates"""
    dh, dw = image_shape
    th, tw = target_shape
    row1, col1 = latlon_to_pixel(coords=tuple(reversed(bbox[0])), transform=transform)
    row2, col2 = latlon_to_pixel(coords=tuple(reversed(bbox[1])), transform=transform)
    row_mean, col_mean = int((row1 + row2) / 2), int((col1 + col2) / 2)
    row_down, row_up = row_mean - th / 2, row_mean + th / 2
    col_left, col_right = col_mean - tw / 2, col_mean + tw / 2
    r1, r2 = clamp_row(row_down, dh), clamp_row(row_up, dh)
    c1, c2 = clamp_col(col_left, dw), clamp_col(col_right, dw)
    h, w = r2 - r1, c2 - c1

    if h < th:
        if r1 == 0:
            r2 = th
        else:
            r1 = dh - th

    if w < tw:
        if c1 == 0:
            c2 = tw
        else:
            c1 = dw - tw

    r1, r2 = clamp_row(r1, dh), clamp_row(r2, dh)
    c1, c2 = clamp_col(c1, dw), clamp_col(c2, dw)
    h, w = r2 - r1, c2 - c1

    if h == th and w == tw:
        coords = ((r1, r2), (c1, c2))
        return coords
    else:
        logging.warning(f"Unable to get coordinates of size ({th}, {tw}) from an image of shape {image_shape}")


def instantiate_dataset(dataset_class, path, **kwargs):
    """Instantiate single dataset"""
    dataset_args = kwargs
    dataset_args["_target_"] = dataset_class
    dataset_args["path"] = path
    if "local_dir" in dataset_args.keys():
        dataset_args.pop("local_dir")
    dataset = instantiate(dataset_args)
    return dataset


def get_dataset(path, region_id="", **kwargs):
    """Download dataset, if necessary. Returns path of dataset."""
    if path.startswith("s3://"):
        if "work_path" in kwargs.keys():
            local_path = os.path.join(kwargs["work_path"], region_id, os.path.basename(path))
            return s3_get_file(path, local_path)
        else:
            raise ValueError("Local directory (local_dir) dataset field required for s3 datasets.")
    else:
        return path


def _compute_datasets_summary(datasets):
    datasets_summary = defaultdict(list)
    for dataset in datasets:
        for k, v in dataset.summary.items():
            datasets_summary[k].append(v)
    return dict(datasets_summary)  # it is evil to hand over defaultdicts to users ;)


class DiSSTLConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.datasets_summary = _compute_datasets_summary(self.datasets)


class DiSSTLListDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_summary = _compute_datasets_summary(self.datasets)


def instantiate_datasets(
    dataset_class, base_path: str = None, satellite: str = None, regions: List[str] = None, concat=True, **kwargs
):
    """Perform pre-initialization functionality for datasets"""
    datasets = []
    for region_id in regions:
        # Get datasets and update paths to local paths, if necessary
        local_region_path = get_dataset(
            os.path.join(base_path, satellite, region_id, "region.hdf5"), region_id=region_id, **kwargs
        )
        dataset = instantiate_dataset(
            dataset_class, local_region_path, satellite=satellite, region_id=region_id, **kwargs
        )
        datasets.append(dataset)

    if concat is True:
        datasets = DiSSTLConcatDataset(datasets)
    else:
        datasets = DiSSTLListDataset(datasets)

    return datasets


def get_date_difference(dates_str: str, date_fmt: str = STD_DATE_FORMAT) -> Dict:
    """Compute the running difference between most recent and oldest dates in teh set.
    Example:
        dates_str = '2017-02-08,2018-01-19,2019-01-09,2019-06-23,2020-06-17,2020-09-25'
        dates_diff = get_date_difference(dates_str)
        print(dates_diff["delta_days"])
        > 1325  # (days)
        i.e., ('2020-09-25' - '2017-02-08').days -> delta_days = 1325
    """
    dates_list = sorted([datetime.strptime(d, date_fmt).date() for d in dates_str.split(",")])
    return dict(delta_days=(dates_list[-1] - dates_list[0]).days, dates=dates_list)


def days_to_ymw(delta_days: int, num_years: int = 4) -> Dict:
    """
    Example:
        delta_days = 1325 (days) = 3.62765 (delta_years) = 43.53182 (delta_months)
         = 189.1563 (delta_weeks)
        returns = {'years': 3.0, 'months': 7.0, 'weeks': 2.0, 'days': 2.0}

        where:
            * years is in [0, model.num_years - 1]],
            * months is in [0, 11], and
            * weeks is in [0, 3]
    """
    delta_years = delta_days / 365.25
    deci_years, years = modf(delta_years)
    # months: 1-12
    deci_months, months = modf(deci_years * 12)
    # weeks: 1-4
    cycle_weeks = deci_months * 4  # 4.345
    # round down weeks i.e., week = week if days < 7 else week + 1 week
    deci_weeks, weeks = modf(cycle_weeks)
    # weeks = years * 52.1429
    # days: 1-7
    _, days = modf(deci_weeks * 7)

    # check cycle bounds and adjust
    if weeks >= 4:  # -> one month
        weeks = 0
        if months + 1 >= 12:  # -> one year
            years += 1
            months = 0
        else:
            months += 1

    # lastly, clip years to exist within pre-defined year bounds
    if years > num_years:
        years = num_years

    return dict(years=int(years), months=int(months), weeks=int(weeks), days=int(days), delta_days=delta_days)
