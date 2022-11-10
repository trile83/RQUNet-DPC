import datetime
import json
import logging
import os
from typing import Dict, Tuple, Optional

import jsonschema
import numpy as np
import torch
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

_logger = logging.getLogger(__name__)


def ymw_to_days(years: int, months: int, weeks: int) -> datetime.timedelta:
    """Convert year/month/week Activity Predictor output to days."""
    days = (years * 365) + (months * 30) + (weeks * 7)
    return datetime.timedelta(days=days)


# TODO improve area computation
def convert_area(km_area: float):
    """Converts area in km2 to 4326 area."""
    km_to_deg_factor = 111  # kilometers
    area = km_area / 1000**2 / km_to_deg_factor**2
    return area


def mask_to_polygons(
    mask: torch.Tensor,
    transform: np.ndarray,
    crs: str,
    threshold: float,
    min_area_threshold: float = 5000.0,
    aoi_geometry: BaseGeometry = None,
) -> Optional[Polygon]:
    """Convert a predicted binary mask to a shapely polygon."""
    transform = Affine(*transform.tolist()[:6])

    mask = (mask > threshold).astype("uint8")

    features = [f for f, v in shapes(mask, mask, transform=transform, connectivity=8)]
    features = transform_geom(src_crs=crs, dst_crs={"init": "EPSG:4326"}, geom=features)

    # TODO If we guarantee that polygons would always come in UTM, we can improve area computation easily
    polygons = [shape(p).buffer(0) for p in features if shape(p).area > convert_area(min_area_threshold)]

    if aoi_geometry is None:  # If we aren't given an AoI, just return the polygon
        return polygons
    else:
        new_polygons = []
        for p in polygons:
            if not p.intersects(aoi_geometry):
                continue

            p = p.intersection(aoi_geometry)
            if p.geom_type == "Polygon":
                p = MultiPolygon([p])

            new_polygons.extend([g.buffer(0) for g in p.geoms])

        return new_polygons


class JsonValidator(object):
    def __init__(self, schema_json):
        """
        :param schema_file: full path to the reference json schema
        """
        self.file = schema_json
        self.schema = load_jfile(schema_json)

    def validate(self, inference_json, target: str = None) -> Tuple[Dict, bool]:
        if target == "file":
            json_object, is_valid = self._validate_file(inference_json)
        elif target == "str":
            json_object, is_valid = self._validate_str(inference_json)
        else:  # check both (file or str)
            _logger.debug(
                f"\tValidator: unclear validation target: '{target}'. Checking for"
                " either valid json file or valid json string!\n"
            )
            json_object, is_valid = self._validate_file(inference_json)
            _logger.debug(f"\t > string is valid = {is_valid}\n")

            if not is_valid:
                json_object, is_valid = self._validate_str(inference_json)
        return json_object, is_valid

    def _validate_file(self, inference_json: os.PathLike):
        """
        Validate the inference json file using the reference json schema template/definition.
        """
        # load json file
        try:
            inference_json = load_jfile(inference_json)
        except ValueError as e:
            _logger.error(f"\t > Invalid json file. Does it exist? It produced error: {e}")
        # verify
        try:
            jsonschema.validate(instance=inference_json, schema=self.schema)
        except jsonschema.exceptions.ValidationError as err:
            msg = f"Invalid json file. Please verify the input \n '{inference_json}'!"
            _logger.error(f"{err}\n > {msg}")
            return False, None
        msg = "Valid input json"
        _logger.info(f"{msg}")
        return inference_json, True

    def _validate_str(self, inference_json: str):
        """
        Validate the inference json file using the reference json schema template/definition.
        """
        # string to json
        try:
            json_object = json.loads(inference_json)
        except ValueError as e:
            _logger.debug(f"\t[DEBUG]Invalid json_string. It produced error: {e}")
        # verify
        try:
            jsonschema.validate(instance=inference_json, schema=self.schema)
        except jsonschema.exceptions.ValidationError as err:
            msg = f"\tInvalid json str! Please verify the input \n '{inference_json}'!"
            _logger.error(f"{err}\n > {msg}")
            return None, False
        msg = "\tValid input json"
        _logger.info(f"{msg}")
        return json_object, True


def parse_request_json(inference_json: str, ref_schema_json: str, owrites: Dict = None):
    """
    :param inference_json: input json file to be valiated by comparing it against the schema.
    :param ref_schema_json: reference json file with the schema definition for data requests & acquisition.
    :param owrites: overwrite arguments from cli
    """
    # the schema validator
    validator = JsonValidator(ref_schema_json)
    json_object, is_valid = validator.validate(inference_json)
    _logger.debug(f"VALIDATION RESPONSE: '{is_valid}'")

    if (owrites is not None) and is_valid:
        for k, v in owrites.items():
            if k in json_object:
                json_object[k] = v

    # Last check
    if is_valid:
        return json_object
    else:
        raise ValueError(f'Unable to verify the provided inference_json.\n"{inference_json}"')


def load_jfile(filename) -> Dict:
    """Helper. Load json file."""
    _logger.debug(f"\tattempting to load file: '{filename}'")
    if not os.path.isfile(filename):
        raise ValueError(f"File: '{filename}' not found. Please verify!")
    else:
        _logger.debug("\t \t >> FOUND!")

        with open(filename, "r") as jfile:
            data = jfile.read()
        return json.loads(data)
