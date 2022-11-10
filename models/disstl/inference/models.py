import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import multiprocessing
from multiprocessing.pool import ThreadPool
from functools import partial

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import numpy as np
from shapely.geometry.base import BaseGeometry

import disstl.datasets.transforms  # noqa F401
from disstl.commons.aws.s3 import s3_get_dir
from disstl.inference import utils
from disstl.models import CPCMultiTemporalSegmentation
import catboost as cb


_logger = logging.getLogger(__name__)

DEFAULT_MODEL_BUNDLE_PATH = Path.home() / ".disstl" / "models" / "latest"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.cuda.amp.autocast()
@torch.no_grad()
def multitemporal_segmentation(
    model: CPCMultiTemporalSegmentation,
    dataloader: DataLoader,
    device: str,
    threshold: float,
    aoi_geometry: BaseGeometry,
    min_area_threshold: float = 5000.0,
):
    if isinstance(dataloader.dataset, torch.utils.data.Subset):
        dataset = dataloader.dataset.dataset
    else:
        dataset = dataloader.dataset

    image_shape = dataset.shape
    num_images = len(dataset.images)

    _logger.info(f"Image shape is {image_shape} with {num_images} images.")
    output_totals = np.zeros([num_images, *image_shape], dtype=np.float16)
    output_counts = np.zeros([num_images, *image_shape], dtype=np.uint16)
    logging_frequency = len(dataloader) // 10
    for i, batch in enumerate(dataloader):
        if i % logging_frequency == 0:
            _logger.info(f"Processed {i+1} MTMS chip batches out of {len(dataloader)}.")

        output = model(batch["x"].to(device))
        output = output.y_pred.softmax(dim=-3)[:, :, 1, :, :].cpu()

        for batch_idx, (window, temporal_window) in enumerate(zip(batch["windows"], batch["temporal_windows"])):
            (rowmin, rowmax), (colmin, colmax) = window
            batch_output = output[batch_idx]
            output_totals[temporal_window, rowmin:rowmax, colmin:colmax] = np.add(
                output_totals[temporal_window, rowmin:rowmax, colmin:colmax], batch_output
            )
            output_counts[temporal_window, rowmin:rowmax, colmin:colmax] += 1

    _logger.info(f"Processed {len(dataloader)} MTMS chip batches out of {len(dataloader)}.")

    # There should be no unprocessed pixels
    assert not np.any(output_counts == 0)
    output_totals /= output_counts

    num_threads = multiprocessing.cpu_count() // 4
    _logger.info(f"Starting vectorization of raster predictions with {num_threads} threads.")
    with ThreadPool(num_threads) as p:
        polygons = list(
            p.imap(
                partial(
                    polygonization_helper,
                    aoi_geometry=aoi_geometry,
                    min_area_threshold=min_area_threshold,
                    transform=dataset.transform,
                    crs=dataset.crs,
                    threshold=threshold,
                ),
                output_totals,
            )
        )
    _logger.info("Vectorization of raster predictions finished.")

    region_output = dict(
        polygons=polygons,
        # See earlier comment about using 0
        images=dataset.images,
        stac_item_ids=dataset.sources,
    )

    return region_output


def polygonization_helper(*args, **kwargs):
    return utils.mask_to_polygons(
        mask=args[0],
        **kwargs,
    )


def load_model(model_path: Path, device=DEFAULT_DEVICE) -> Tuple[nn.Module, DictConfig]:
    """
    Returns the a trained model.

    :param model_path: path to model directory containing config.yml, weights.pt
    :param device: cuda device to use (if is available)
    :return: nn.Module
    """
    device = device if torch.cuda.is_available() else "cpu"
    weights_path = model_path / "weights.pt"
    config_path = model_path / "config.yaml"
    cfg = OmegaConf.load(config_path)
    if "paradigm" in cfg:
        # this must be a current model (after !224)
        cfg_model = cfg.paradigm.model
        spectral_indices = cfg.satellite.spectral_indices
        band_names = cfg.satellite.band_names
        num_channels = len(band_names) + len(spectral_indices)
        cfg.paradigm.model.channels = num_channels
    else:
        # this must be an old model (prior to !224), use the old config key format
        cfg_model = cfg.model
        spectral_indices = cfg.data.spectral_indices
        band_names = cfg.data.band_names
        num_channels = len(band_names) + len(spectral_indices)
        cfg.model.channels = num_channels

    # Override channels and cpc_state_dict_path params
    cfg_model["channels"] = num_channels
    if "cpc_state_dict_path" in cfg_model:
        cfg_model["cpc_state_dict_path"] = None
    if "pretrained" in cfg_model:
        # Disable pre-trained weight loading during inference
        cfg_model["pretrained"] = False
    # Load model and pretrained weights
    model: nn.Module = instantiate(cfg_model)
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(dict(model.named_parameters()).keys())

    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()
    return model, cfg


def load_cb_model(model_path: Path):
    cb_model_path = model_path / "model.cb"
    if not cb_model_path.exists():
        return None, None

    files = os.listdir(model_path)
    model_id = None
    for file in files:
        if file.endswith(".json"):
            model_id = file.replace(".json", "")

    model_params = {
        "objective": "RMSE",
        "depth": 12,
        "boosting_type": "Plain",
        "bootstrap_type": "MVS",
        "learning_rate": 0.27039877125764233,
        "iterations": 1500,
    }
    model = cb.CatBoostRegressor(**model_params)
    model.load_model(cb_model_path)

    return model, model_id


def load_model_ckpt(model_path: Path, ckpt_type: str, device=DEFAULT_DEVICE) -> Tuple[nn.Module, DictConfig]:
    """
    Returns the a trained model from a pytorch lightning checkpoint.

    :param model_path: path to model directory containing config.yml, weights.pt
    :param device: cuda device to use (if is available)
    :return: nn.Module
    """
    device = device if torch.cuda.is_available() else "cpu"
    config_path = model_path / "config.yaml"
    cfg = OmegaConf.load(config_path)
    if "paradigm" in cfg:
        # this must be a current model (after !224)
        cfg_model = cfg.paradigm.model
        spectral_indices = cfg.satellite.spectral_indices
        band_names = cfg.satellite.band_names
    else:
        # this must be an old model (prior to !224), use the old config key format
        cfg_model = cfg.model
        spectral_indices = cfg.data.spectral_indices
        band_names = cfg.data.band_names

    if spectral_indices is None:
        spectral_indices = []
    if band_names is None:
        band_names = []
    num_channels = len(band_names) + len(spectral_indices)
    if "paradigm" in cfg:
        cfg.paradigm.model.channels = num_channels
    else:
        cfg.model.channels = num_channels

    # Override channels and cpc_state_dict_path params
    cfg_model["channels"] = num_channels
    if "cpc_state_dict_path" in cfg_model:
        cfg_model["cpc_state_dict_path"] = None
    if "pretrained" in cfg_model:
        # Disable pre-trained weight loading during inference
        cfg_model["pretrained"] = False
    # Load model and pretrained weights
    model: nn.Module = instantiate(cfg_model)
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(dict(model.named_parameters()).keys())

    state_dict = torch.load(model_path / f"{ckpt_type}.ckpt", map_location=torch.device(device))["state_dict"]
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    if hasattr(model, "cpc"):
        model.cpc.inference = True
    return model, cfg


def stage_model(model_loc: Optional[Union[str, Path]], work_path: Path) -> Path:
    """
    If s3 model_loc, download model from s3, and unzip if necessary.

    :param model_loc:  s3 uri or local file path
    :param work_path: Where model data should be downloaded to, in case that model_loc is s3://... uri
    :return: Path to local model directory, or default model path
    """

    if isinstance(model_loc, Path):
        return model_loc

    model_path = DEFAULT_MODEL_BUNDLE_PATH

    if isinstance(model_loc, str):
        if model_loc.startswith("s3://"):
            model_path = work_path / "model"
            s3_get_dir(model_loc, model_path)
        else:
            model_path = Path(model_loc).expanduser()

    return model_path.expanduser()
