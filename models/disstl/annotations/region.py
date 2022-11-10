""" region.py defines the Region class for storing region-specific metadata """
from __future__ import annotations

import json
import logging
from botocore.exceptions import ClientError
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
import catboost as cb
from geopandas import gpd

from disstl.annotations.enums import Activity
from disstl.annotations.site import Site
from disstl.annotations.site_observation import SiteObservation
from disstl.annotations.utils import as_date, as_date_str, orient_geometry
from disstl.commons.aws.s3 import s3_get_json
from disstl.standards import STD_DATE_FORMAT
from disstl.utils.debug import timing

_logger = logging.getLogger(__name__)


def to_box(x: Tuple[Tuple[Any, Any], Tuple[Any, Any]]) -> Polygon:
    return box(minx=x[0][0], miny=x[1][0], maxx=x[0][1], maxy=x[1][1])


def geometry_to_grid(row, lat_grid=60, lon_grid=120):
    """Convert lat lon to grid indices

    Args:
        row (_type_): row from a geopandas dataframe
        lat_grid (int, optional): number of latitude grid cells. Defaults to 60.
        lon_grid (int, optional): number of longitude grid cells. Defaults to 120.

    Returns:
        tuple(grid_idx_lat, grid_idx_lon): the grid indices for lat and lon, respectively
    """
    geom = row["geometry"]
    # y = lat, x = lon
    lat, lon = geom.centroid.y, geom.centroid.x
    grid_idx_lat = (lat % 180) // (180 / lat_grid)
    grid_idx_lon = (lon % 360) // (360 / lon_grid)
    return grid_idx_lat, grid_idx_lon


def get_next_phase(current_phase):
    if current_phase == Activity.SITE_PREPARATION:
        return Activity.ACTIVE_CONSTRUCTION
    elif current_phase == Activity.ACTIVE_CONSTRUCTION:
        return Activity.POST_CONSTRUCTION
    elif current_phase == Activity.POST_CONSTRUCTION:
        return Activity.POST_CONSTRUCTION


class Region:
    """Region contains region-specific metadata"""

    def __init__(  # TODO determine which fields are not optional.
        self,
        geometry: BaseGeometry,
        region_id: Optional[str] = None,
        version: Optional[str] = None,
        mgrs: Optional[str] = None,
        model_content: Optional[str] = "proposed",
        originator: Optional[str] = "bla",
        comments: Optional[str] = None,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
        sites: Optional[Dict] = None,
    ):
        self.region_id = region_id
        self.version = version
        self.mgrs = mgrs
        self.model_content = model_content
        self.originator = originator
        self.comments = comments
        self.start_date = as_date(start_date)
        self.end_date = as_date(end_date)
        self.geometry = geometry
        self._sites = sites if sites is not None else {}

    def __eq__(self, other):
        return (type(other) is type(self)) and (self.__dict__ == other.__dict__)

    @property
    def sites(self):
        return self._sites.values()

    @property
    def site_ids(self):
        return self._sites.keys()

    def get_site(self, site_id):
        return self._sites[site_id]

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        path = Path(file)
        with open(path, "r") as fp:
            region_json = json.load(fp)
        region = cls.from_geojson(region_json)

        for site_id in region.site_ids:
            site_path = path.parent.parent / "site_models" / f"{site_id}_BLA.geojson"
            if not site_path.is_file():
                site_path = path.parent.parent / "site_models" / f"{site_id}.geojson"
            with open(site_path, "r") as fp:
                site_json = json.load(fp)
            region.update_site_from_geojson(site_json)
        return region

    @classmethod
    @timing(start_msg="fetching region annotations", end_msg="region annotations fetched", logger=_logger)
    def from_s3_annotations(cls, region_id: str) -> Region:
        s3_region_folder = (
            "s3://bksy-smart-prf-train/data/regions/annotated/region_models/"
            if "BLA" in region_id
            else "s3://smart-imagery/annotations/region_models/"
        )
        region_json = s3_get_json(f"{s3_region_folder}{region_id}.geojson")
        region = Region.from_geojson(region_json)
        for site_id in region.site_ids:
            s3_site_folder = (
                "s3://bksy-smart-prf-train/data/regions/annotated/site_models/"
                if "BLA" in site_id
                else "s3://smart-imagery/annotations/site_models/"
            )
            try:
                site_json = s3_get_json(f"{s3_site_folder}{site_id}.geojson")
            except ClientError as ce:
                if ce.response["Error"]["Code"] == "NoSuchKey":
                    _logger.warn(f"Site model geojson {s3_site_folder}{site_id}.geojson was not found in S3.")
                    continue
                raise
            region.update_site_from_geojson(site_json)
        return region

    @classmethod
    def from_geojson(cls, geo_json: Union[str, Dict]):
        """Builds this region"""
        if type(geo_json) is str:
            geo_json_dict = json.loads(geo_json)
        else:
            geo_json_dict = geo_json
        regions = list(filter(lambda feature: feature["properties"]["type"] == "region", geo_json_dict["features"]))
        if len(regions) != 1:
            raise ValueError("more than one region defined in the GeoDataFrame")
        region = regions[0]
        region_props = region["properties"]
        sites = {}
        for site in filter(lambda feature: feature["properties"]["type"] == "site_summary", geo_json_dict["features"]):
            sites[site["properties"]["site_id"]] = Site.from_region_model_geojson_item(site)
        return cls(
            shape(region["geometry"]),
            region_id=region_props.get("region_id"),
            version=region_props.get("version"),
            mgrs=region_props.get("mgrs"),
            model_content=region_props.get("model_content"),
            originator=region_props.get("originator"),
            comments=region_props.get("comments"),
            start_date=region_props.get("start_date"),
            end_date=region_props.get("end_date"),
            sites=sites,
        )

    @classmethod
    def from_inference_result(
        cls,
        result: List[Dict],
        inference_type: str,
        input_region_or_geom: Optional[Union[Region, BaseGeometry]] = None,
        indices: Optional[List[int]] = None,
        sensor_name: Optional[str] = None,
        model_id: Optional[str] = None,
        cb_model: Optional[cb.CatBoostRegressor] = None,
        cb_model_id: Optional[str] = None,
        disstl_parameters: Optional[dict] = None,
        min_num_observations_per_site: int = 10,
        run_id: Optional[str] = None,
    ):
        num_images = len(result["images"])
        start_date, end_date = result["images"][0], result["images"][-1]
        region_id = getattr(input_region_or_geom, "region_id", "region")

        inference_model_ids = {}
        input_sites = list(getattr(input_region_or_geom, "sites", []))
        if input_sites and input_sites[0].misc_info is not None:
            inference_model_ids = {
                key: value
                for key, value in input_sites[0].misc_info.items()
                if ("model_id" in key or key == "model_package_id" or key == "run_id")
            }
            if "cb_model_id" in inference_model_ids:
                del inference_model_ids["cb_model_id"]
        if run_id is not None and inference_model_ids.get("run_id") is None:
            inference_model_ids["model_package_id"] = run_id
            inference_model_ids["run_id"] = run_id

        # Compute region geometries across images
        region_polygons = [p for polys in result["polygons"] for p in polys]
        region_geometry = unary_union(region_polygons)
        region_geometry = orient_geometry(region_geometry)

        sites = []
        for i, site_geometry in enumerate(region_geometry.geoms):
            observations = []
            for image_index, image in enumerate(result["images"]):
                if all(not p.intersects(site_geometry) for p in result["polygons"][image_index]):
                    continue  # no image polygon intersects site

                if "stac_item_ids" in result:
                    stac_item_ids = ",".join(result["stac_item_ids"][image_index])
                else:
                    stac_item_ids = None

                site_index = i

                if inference_type == "seg_nobas":
                    score = 1.0  # TODO extract from segmentation polygons

                temp_coords = [p for p in result["polygons"][image_index] if p.intersects(site_geometry)]
                coords = unary_union(temp_coords)
                coords = orient_geometry(coords)

                if coords and not coords.is_empty and coords.is_valid and coords.area > 0:
                    obs_misc_info = {
                        "scores": None,
                        f"{inference_type}_model_id": model_id,
                        "cb_model_id": cb_model_id,
                        **inference_model_ids,
                    }
                    if disstl_parameters:
                        obs_misc_info["disstl_parameters"] = disstl_parameters

                    observations.append(
                        SiteObservation(
                            coords,
                            observation_date=image,
                            source=stac_item_ids,
                            sensor_name=sensor_name,
                            current_phase=None,
                            score=score,
                            misc_info=obs_misc_info,
                        )
                    )

            min_num_observations_per_site = min(num_images // 10, min_num_observations_per_site)
            if not observations or len(observations) <= min_num_observations_per_site:
                continue

            if isinstance(input_region_or_geom, Region):
                site_id = f"{input_region_or_geom.region_id}_{str(site_index+1).zfill(4)}"
            else:
                site_id = f"site_{str(site_index+1).zfill(4)}"

            if inference_type == "seg_nobas":
                if cb_model is not None:
                    observations_for_gpd = []
                    for observation in observations:
                        observations_for_gpd.append(
                            {"observation_date": observation.observation_date, "geometry": observation.geometry}
                        )

                    observations_df = gpd.GeoDataFrame(observations_for_gpd, crs="epsg:4326")
                    observations_df["area"] = observations_df.to_crs({"proj": "cea"})["geometry"].map(
                        lambda p: p.area / 10**6
                    )
                    observations_df["rectangleness"] = observations_df.to_crs({"proj": "cea"})["geometry"].map(
                        lambda p: p.area / p.minimum_rotated_rectangle.area
                    )
                    observations_df[["grid_idx_lat", "grid_idx_lon"]] = observations_df.apply(
                        geometry_to_grid, axis=1, result_type="expand"
                    )
                    observations_df["delta_from_construction_start"] = (
                        observations_df["observation_date"] - observations[0].observation_date
                    )

                    input_cols = [
                        "area",
                        "rectangleness",
                        "grid_idx_lat",
                        "grid_idx_lon",
                        "delta_from_construction_start",
                    ]
                    input_df = observations_df[input_cols]
                    predictions = cb_model.predict(input_df)

                    current_phase = Activity.SITE_PREPARATION
                    phase_transition_days = 243  # mean from labeled data
                    for pred_idx, prediction in enumerate(predictions):
                        if input_df["delta_from_construction_start"].iloc[pred_idx].days > phase_transition_days:
                            current_phase = get_next_phase(current_phase)
                            phase_transition_days = float("inf")
                        else:
                            if prediction >= 0:
                                next_transition_days = (
                                    prediction + input_df["delta_from_construction_start"].iloc[pred_idx].days
                                )
                                phase_transition_days = (
                                    next_transition_days
                                    if phase_transition_days == float("inf")
                                    else (phase_transition_days + next_transition_days) / 2
                                )

                        observations[pred_idx].current_phase = (current_phase,)

                score = 1.0
                if phase_transition_days == float("inf"):
                    phase_transition_days = max(878 - input_df["delta_from_construction_start"].iloc[-1].days, 0)

                predicted_phase_transition_date = observations[-1].observation_date + timedelta(
                    days=phase_transition_days
                )

            site_misc_info = {
                f"{inference_type}_model_id": model_id if model_id is not None else None,
                **inference_model_ids,
            }
            if disstl_parameters:
                site_misc_info["disstl_parameters"] = disstl_parameters

            sites.append(
                Site(
                    site_geometry,
                    region_id=region_id,
                    site_id=site_id,
                    version=getattr(input_region_or_geom, "version", None),
                    mgrs=getattr(input_region_or_geom, "mgrs", None),
                    start_date=start_date,
                    end_date=end_date,
                    score=score,
                    predicted_phase_transition=get_next_phase(observations[-1].current_phase),
                    predicted_phase_transition_date=predicted_phase_transition_date,
                    misc_info=site_misc_info,
                    observations=observations,
                )
            )

        if isinstance(input_region_or_geom, Region):
            geom = input_region_or_geom.geometry
        elif isinstance(input_region_or_geom, BaseGeometry):
            geom = input_region_or_geom
        else:
            geom = sites[0].geometry

        region = cls(
            region_id=region_id,
            version=getattr(input_region_or_geom, "version", None),
            mgrs=getattr(input_region_or_geom, "mgrs", None),
            start_date=getattr(input_region_or_geom, "start_date", start_date),
            end_date=getattr(input_region_or_geom, "end_date", end_date),
            geometry=geom,
            sites={site.site_id: site for site in sites},
        )
        return region

    def to_geojson(self):
        output = {
            "type": "FeatureCollection",
            "features": [self.to_geojson_item()] + [site.to_region_model_geojson_item() for site in self.sites],
        }
        return output

    def to_geojson_item(self):
        return {
            "type": "Feature",
            "properties": {
                "type": "region",
                "region_id": self.region_id,
                "version": self.version,
                "mgrs": self.mgrs,
                "model_content": self.model_content,
                "originator": self.originator,
                "comments": self.comments,
                "start_date": as_date_str(self.start_date),
                "end_date": as_date_str(self.end_date),
            },
            "geometry": mapping(self.geometry),
        }

    def update_site_from_geojson(self, site_model_geojson: Union[str, Dict]):
        """adds a site observation to a site contained in this region"""
        site_id = site_model_geojson["features"][0]["properties"]["site_id"]
        self._sites[site_id] = Site.from_geojson(site_model_geojson)

    def get_geometries_for_date(self, slice_date: date) -> List[Tuple[Site, BaseGeometry, Activity]]:
        output = []
        for _, site in self._sites.items():
            observation = site.get_observation_for_date(slice_date)
            if observation is not None:
                geometries = observation.geometry.geoms
                for geo_index, geom in enumerate(geometries):
                    phase_index = geo_index
                    # some observations have fewer phase classifications than geometries. In these cases
                    # we just assume that the "last" phase listed applies to every geometry.
                    if geo_index >= len(observation.current_phase):
                        phase_index = len(observation.current_phase) - 1
                    output.append((site, geom, observation.current_phase[phase_index]))
        return output

    def get_site_observations_with_date(self, slice_date: date) -> List[Tuple[Site, SiteObservation]]:
        output = []
        for site_id, site in self._sites.items():
            observation = site.get_observation_with_observation_date(slice_date)
            if observation is not None:
                output.append((site, observation))
        return output

    @property
    def period_str(self):
        """returns the time period associated with this region as a string"""
        return f"{self.start_date.strftime(STD_DATE_FORMAT)}/{self.end_date.strftime(STD_DATE_FORMAT)}"
