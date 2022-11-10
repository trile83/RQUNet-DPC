import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader

from disstl.annotations.region import Region
from disstl.cli_opts import GlobalOpts
from disstl.inference import models
from disstl.inference.datasets import InferenceHDF5Dataset, collate_fn, load_indices, load_transforms
from disstl.utils.debug import timing

_logger = logging.getLogger(__name__)


@timing(start_msg="Running full inference run", end_msg="Full inference run completed", logger=_logger)
def run_inference(
    global_opts: GlobalOpts,
    device: str,
    local_cube_path: str,
    output_path: Optional[str],
    inference_type: Optional[str],
    chip_shape: int,
    batch_size: int,
    num_workers: int,
    num_prefetch: int,
    threshold_seg: float,
    min_area_threshold: float,
    stride: int,
    include_parameters: bool,
    disstl_parameters: dict,
    ckpt_type: str,
    model_loc: Optional[Union[str, Path]] = None,
    seg_model_loc: Optional[Union[str, Path]] = None,
    cb_model_loc: Optional[Union[str, Path]] = None,
    region: Optional[Union[Region, BaseGeometry]] = None,
    run_id: Optional[str] = None,
) -> Tuple[Path, Region]:

    _logger.info(f"run_inference params: {json.dumps(locals(), default=str)}")

    if inference_type == "seg_nobas":
        results_path, final_region, _ = inference(
            global_opts,
            device,
            local_cube_path,
            output_path,
            "seg_nobas",
            include_parameters,
            disstl_parameters,
            ckpt_type,
            chip_shape=(chip_shape, chip_shape),
            batch_size=batch_size,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            threshold_seg=threshold_seg,
            min_area_threshold=min_area_threshold,
            stride=(stride, stride) if stride != -1 else None,
            model_loc=model_loc,
            seg_model_loc=seg_model_loc,
            cb_model_loc=cb_model_loc,
            region=region,
            indices=None,
            write_output=True,
            run_id=run_id,
        )
    else:
        raise NotImplementedError(f"Inference type {inference_type} is not implemented.")
    return results_path, final_region


@timing(start_msg="Running step inference run", end_msg="Step inference run completed", logger=_logger)
@torch.no_grad()
def inference(
    opts: GlobalOpts,
    device: str,
    hdf5_path: str,
    output_path: Optional[str],
    inference_type: str,
    include_parameters: bool,
    disstl_parameters: dict,
    ckpt_type: str,
    chip_shape: Tuple[int, int] = (32, 32),
    batch_size: int = 1,
    num_workers: int = 1,
    num_prefetch: int = 1,
    threshold_seg: float = 0.1,
    min_area_threshold: float = 5000.0,
    stride: Tuple[int, int] = None,
    model_loc: Optional[Union[str, Path]] = None,
    seg_model_loc: Optional[Union[str, Path]] = None,
    cb_model_loc: Optional[Union[str, Path]] = None,
    region: Optional[Union[Region, BaseGeometry]] = None,
    indices: Optional[List[int]] = None,
    write_output: bool = True,
    run_id: str = None,
) -> Tuple[Path, Region, List[int]]:
    if device == "cuda" and not torch.cuda.is_available():
        "cuda device unavailable, using cpu"

    if inference_type == "seg_nobas":
        if seg_model_loc is None:
            seg_model_path = models.stage_model(model_loc, opts.work_path) / "segmentation" / "binary"
        else:
            seg_model_path = models.stage_model(seg_model_loc, opts.work_path) / "segmentation" / "binary"
        if cb_model_loc is None:
            cb_model_path = models.stage_model(model_loc, opts.work_path) / "cb_activity"
        else:
            cb_model_path = models.stage_model(cb_model_loc, opts.work_path) / "cb_activity"

        cb_model, cb_model_id = models.load_cb_model(cb_model_path)
        if isinstance(region, BaseGeometry):
            aoi_geometry = region
        else:
            aoi_geometry = region.geometry if region is not None else None

        inference_fn = partial(
            models.multitemporal_segmentation,
            threshold=threshold_seg,
            aoi_geometry=aoi_geometry,
            min_area_threshold=min_area_threshold,
        )
        model, cfg = models.load_model_ckpt(seg_model_path, ckpt_type, device=device)
    else:
        raise ValueError("Unknown inference run type")

    if "paradigm" in cfg:
        # this must be a current model (after !224)
        spectral_indices = cfg.satellite.spectral_indices
        band_names = cfg.satellite.band_names
    else:
        # this must be an old model (prior to !224), use the old config key format
        spectral_indices = cfg.data.spectral_indices
        band_names = cfg.data.band_names
    if spectral_indices is None:
        spectral_indices = []
    if band_names is None:
        band_names = []

    spectral_indices = load_indices(spectral_indices=spectral_indices, bands=band_names)
    transforms = load_transforms(input_shape=chip_shape, transforms=instantiate(cfg.load_transforms))
    transforms = T.Compose([transforms, spectral_indices])

    _logger.info(f"In {inference_type} mode:")
    dataset = InferenceHDF5Dataset(
        path=hdf5_path, chip_shape=chip_shape, stride=stride, transforms=transforms, band_names=band_names
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=num_prefetch,
        collate_fn=collate_fn,
    )

    output = inference_fn(model, dataloader, device=device)
    with h5py.File(hdf5_path, "r") as f:
        sensor_name = json.loads(f["metadata"]["cube"][0].decode("utf-8")).get("sensor_name")

    if run_id is None and model_loc is not None:
        if model_loc.endswith("/"):
            run_id = model_loc.split("/")[-2]
        else:
            run_id = model_loc.split("/")[-1]

    if "experiment" in cfg and "default_run_name" in cfg.experiment:
        model_id = cfg.experiment.default_run_name
    else:
        model_id = None

    region = Region.from_inference_result(
        output,
        inference_type,
        input_region_or_geom=region,
        indices=indices,
        sensor_name=sensor_name,
        model_id=model_id,
        cb_model=cb_model,
        cb_model_id=cb_model_id,
        disstl_parameters=disstl_parameters if include_parameters else None,
        run_id=run_id,
    )

    if write_output:
        region_output_path = output_path / "region_models"
        site_output_path = output_path / "site_models"
        for directory in [region_output_path, site_output_path]:
            directory.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(region_output_path, f"{region.region_id}_BLA.geojson"), "w") as region_file:
            region_file.write(json.dumps(region.to_geojson()))

        for site in region.sites:
            with open(os.path.join(site_output_path, f"{site.site_id}_BLA.geojson"), "w") as site_file:
                site_file.write(json.dumps(site.to_geojson()))

    return Path(output_path), region, indices
