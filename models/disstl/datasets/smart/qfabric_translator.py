#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translates qfabric annotations to smart annotations (site_models and region_models)
@author: ctorres <ctorres@blacksy.com>
"""
# built-in
import os
import ast
import sys
import json
import logging as log
from datetime import datetime
from typing import List, Dict, Tuple

# data
import numpy as np
import pandas as pd
import geopandas as gpd
from natsort import natsorted
from shapely.geometry import Polygon

# display and plotting
import contextily as ctx

log.basicConfig(stream=sys.stdout, level=log.INFO)


ANNOT_VERSION = "0.6.0"
SOURCE = "smart-qfabric"
CHANGE_TYPES = [
    "Industrial",
    "Mega Projects",
    # "Commercial",
]


# MISC METHODS
def plot_on_map(df: pd.DataFrame, epsg: int = 3758, title: str = None, src_column: str = "geometry"):
    color = df.color.tolist() if "color" in df else "red"

    alpha = df.alpha.tolist() if "alpha" in df else 0.5

    ax = df[src_column].to_crs(epsg=epsg).plot(figsize=(10, 10), alpha=alpha, facecolor=color, edgecolor="k")
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)


def confirm_dir(directory: os.PathLike = "temp"):
    """Convenience.
    Checks if a directory exists exists, else creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_polygon(coords, polygon_name: str = "temp", crs: Dict = {"init": "epsg:4326"}) -> pd.DataFrame:
    """Create a polygon from coordinates.
    crs = {'init' :'epsg:3857'}  #  metric
    crs = {'init' :'epsg:4326'}  # lat lon
    """
    polygon = Polygon(coords)
    gdf = gpd.GeoDataFrame(crs=crs)
    gdf.loc[0, "name"] = polygon_name
    gdf.loc[0, "geometry"] = polygon
    return gdf


def get_region_bounds(region: pd.DataFrame) -> Polygon:
    """Helper.
    Extract the outter perimeter lat-lon coordinates
    and use it to build/return  a shapely.Polygon object.
    """
    # the AOI perimeter: total_bounds.
    lat0, lon0, lat1, lon1 = region.total_bounds
    # counter-clockwise order: lon-lat
    coords = [
        (lat0, lon0),  # topleft
        (lat0, lon1),  # botleft
        (lat1, lon1),  # botright
        (lat1, lon0),  # topright
        (lat0, lon0),  # topleft
    ]
    # poly_df = create_polygon(coords, f"region_{region_index}", crs=crs)
    return Polygon(coords)


# QFABRIC -> SMART MAPPING
# From QFabric Phase Label -> SMART Phase Label
QFABRIC2SMART_LABEL_MAP = {
    # Unlabeled: None
    # -- site_preparation
    "Excavation": "Site Preparation",
    "Land Cleared": "Site Preparation",
    "Materials Introduced": "Site Preparation",
    "Materials Dumped": "Site Preparation",
    # -- active_construction
    "Construction Started": "Active Construction",
    "Construction Midway": "Active Construction",
    # -- post_construction
    "Construction Done": "Post Construction",
    "Operational": "Post Construction",
    # -- Unknown: None
    # -- no_activity
    "Greenland": "No Activity",
    "Prior Construction": "No Activity",
}


def qfabric_to_smart_site_translator(
    site: pd.DataFrame, activity_dates_dict: Dict, qregion_id: int, qsite_id: int, disp: bool = False
) -> gpd.GeoDataFrame:
    """Helper.
    Translates Qfrabric site (row) to SMART site (file).
    """
    observation_date_list = []
    current_phase_list = []
    for k, v in activity_dates_dict.items():
        nrows = len(v)
        observation_date_list = list(v.keys())
        current_phase_list = list(v.values())

    sorted_observation_date_list = sorted(observation_date_list, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    sorted_observation_date_list

    nrows = len(observation_date_list)

    # polygon
    polygon = site["geometry"]
    hlen = len(polygon.geometry.tolist())

    # HEADER: DATA
    qsite_first_row = {
        "type": ["site"] * hlen,
        "region_id": [qregion_id] * hlen,
        "site_id": [qsite_id] * hlen,
        "version": [ANNOT_VERSION] * hlen,
        "status": ["system_confirmed"] * hlen,
        "mgrs": [None] * hlen,
        "score": [1.0] * hlen,
        "start_date": [sorted_observation_date_list[0]] * hlen,
        "end_date": [sorted_observation_date_list[-1]] * hlen,
        "model_content": ["annotation"] * hlen,
        "originator": ["bla"] * hlen,
        "validated": [None] * hlen,
        "observation_date": [None] * hlen,
        "source": [SOURCE] * hlen,
        "sensor_name": ["pending"] * hlen,
        "current_phase": [None] * hlen,
        "is_occluded": [None] * hlen,
        "is_site_boundary": [None] * hlen,
        "geometry": polygon.geometry.tolist(),
        "qfabric_phase": [None] * hlen,
        "color": ["white"] * hlen,
        "alpha": [0.25] * hlen,
    }
    if disp:
        print("FIRST ROW")
        for k, v in qsite_first_row.items():
            print(f"key: '{k}' has {len(v)} - values")

    site_header_df = pd.DataFrame(data=qsite_first_row)
    # current_phase_list
    polygons = polygon.geometry.tolist() * nrows

    # BODY: DATA
    qsite_rows = {
        "type": ["observation"] * nrows * hlen,
        "region_id": [None] * nrows * hlen,
        "site_id": [qsite_id] * nrows * hlen,
        "version": [None] * nrows * hlen,
        "status": [None] * nrows * hlen,
        "mgrs": [None] * nrows * hlen,
        "score": [1.0] * nrows * hlen,
        "start_date": [None] * nrows * hlen,
        "end_date": [None] * nrows * hlen,
        "model_content": [None] * nrows * hlen,
        "originator": ["bla"] * nrows * hlen,
        "validated": [None] * nrows * hlen,
        "observation_date": observation_date_list * hlen,
        "source": [SOURCE] * nrows * hlen,
        "sensor_name": ["pending"] * nrows * hlen,
        "current_phase": current_phase_list * hlen,
        "is_occluded": ["pending"] * nrows * hlen,
        "is_site_boundary": [True] * nrows * hlen,
        "geometry": polygons,
        "qfabric_phase": current_phase_list * hlen,
        "color": ["blue"] * nrows * hlen,
        "alpha": [0.25] * nrows * hlen,
    }
    if disp:
        print("BODY ROWs")
        for k, v in qsite_rows.items():
            print(f"key: '{k}' has {len(v)} - values")

    site_rows_df = pd.DataFrame(data=qsite_rows)

    # PUTTING ALL TOGETHER
    df = pd.concat([site_header_df, site_rows_df])
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf.reset_index(inplace=True, drop=True)

    # THE ACTIVITY MAP
    gdf["current_phase"] = gdf["current_phase"].map(QFABRIC2SMART_LABEL_MAP)
    return gdf


def qfabric_to_smart_region_translator(
    qregion: gpd.GeoDataFrame, qregion_id: str, misc_info: str = None
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Converts qfabric_row to smart region_model"""
    # BASIC PARAMS
    qregion_bounds = get_region_bounds(qregion)
    labels = qregion.label.tolist()
    nrows = len(qregion)
    qregion_dates, activity_dates_dict = qfabric_operating_dates(qregion, labels=labels)

    # HEADER
    qregion_first_row = {
        "type": ["region"],
        "region_id": [f"{qregion_id}"],
        "version": [ANNOT_VERSION],
        "mgrs": [None],
        "start_date": [qregion_dates["region"][0]],
        "end_date": [qregion_dates["region"][1]],
        "originator": ["bla"],
        "model_content": ["annotation"],
        "status": [None],
        "site_id": [None],
        "score": [np.NaN],
        "validated": [None],
        "geometry": [qregion_bounds],
        "qfabric_urban_type": [(";").join(qregion["urban_type"].unique().tolist())],
        "qfabric_geography_type": [(";").join(qregion["geography_type"].unique().tolist())],
        "qfabric_change_type": [(";").join(CHANGE_TYPES)],
        "color": ["red"],
        "alpha": [0.25],
        "comments": [misc_info],
    }
    header_df = pd.DataFrame(data=qregion_first_row)

    # BODY
    qregion_rows = {
        "type": ["site_summary"] * nrows,
        "region_id": [None] * nrows,
        "version": [ANNOT_VERSION] * nrows,
        "mgrs": [None] * nrows,
        "start_date": [qregion_dates[f"site_{str(i).zfill(4)}"][0] for i in labels],
        "end_date": [qregion_dates[f"site_{str(i).zfill(4)}"][1] for i in labels],
        "originator": ["bla"] * nrows,
        "model_content": ["annotation"] * nrows,
        "status": ["pending"] * nrows,
        "site_id": [f"{qregion_id}_{str(r).zfill(4)}" for r in labels],
        "score": [1.0] * nrows,
        "validated": [None] * nrows,
        "geometry": qregion["geometry"],
        "qfabric_urban_type": qregion["urban_type"],
        "qfabric_geography_type": qregion["geography_type"],
        "qfabric_change_type": qregion["change_type"],
        "color": ["blue"] * nrows,
        "alpha": [0.5] * nrows,
    }
    rows_df = pd.DataFrame(data=qregion_rows)

    # PUT IT ALL TOGETHER:
    df = pd.concat([header_df, rows_df])
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    gdf.reset_index(inplace=True, drop=True)
    return gdf, activity_dates_dict


def qfabric_activity_filter(df: pd.DataFrame, filter1: List = CHANGE_TYPES, filter2: int = 0) -> pd.DataFrame:
    """QFabric Helper.
    Apply filters to the qfabric region dataframe.
    """
    # filter 1: specific construction types.
    df = df[df["change_type"].isin(filter1)]
    # filter 2: select date-annotated activities.
    df = df[df["change_status"].str.len() > filter2]
    return df


def qfabric_operating_dates(qregion: pd.DataFrame, src_column: str = "change_status", labels: List = None) -> Dict:
    """QFabric Helper.
    Prses and organizes activities, dates, and date ranges (observation active dates).
    """
    dates = []
    active_dates = dict()
    activity_dict = dict()
    # activity-date and activity-label info (i.e., qfabric: json-str column)
    area_dates = qregion[src_column].values.tolist()
    labels = [r for r in range(len(area_dates))] if labels is None else labels

    for a, area_date in zip(labels, area_dates):
        temp_dict = ast.literal_eval(json.loads(area_date))
        temp_dict_keys = [key for key in temp_dict.keys() if key != "--"]
        activity_dates = [datetime.strptime(dates, "%d-%m-%Y").strftime("%Y-%m-%d") for dates in temp_dict_keys]
        activity_labels = temp_dict.values()
        # area dates:
        area_date_keys = [key for key in ast.literal_eval(json.loads(area_date)).keys() if key != "--"]

        area_dates = sorted(area_date_keys, key=lambda x: datetime.strptime(x, "%d-%m-%Y"))
        area_dates = [datetime.strptime(dates, "%d-%m-%Y").strftime("%Y-%m-%d") for dates in area_dates]
        # site observation dates:
        start_date, end_date = area_dates[0], area_dates[-1]
        active_dates[f"site_{str(a).zfill(4)}"] = (start_date, end_date)
        # the sequence of observations-dates
        dates.extend(area_dates)  # single list
        activity_dict[a] = {k: v for k, v in zip(activity_dates, activity_labels)}
    sorted_dates = sorted(dates)
    active_dates["region"] = (sorted_dates[0], sorted_dates[-1])

    return active_dates, activity_dict


# MAIN CLASS
class QFABTranslator:
    """Class that loads and parses QFabric annotations and translates
    their contents to SMART region_models and site_models formats.
    """

    def __init__(
        self,
        input_directory: os.PathLike,
        ext: str = ".geojson",
        tag: str = "TEMP_R",
        region_id_map_csvfile: os.PathLike = None,
    ):
        self.input_directory = input_directory
        self.ext = ext
        self.columns: List = []
        self.filepaths: List = self.find_files()
        self.ntotal_paths = len(self.filepaths)
        self.failed_jsons: List = []
        self.output_directory: os.PathLike = None
        self.tag: str = tag
        self.activity_label_map: Dict = QFABRIC2SMART_LABEL_MAP
        self.translated_site_count = 0
        self.translated_region_count = 0
        self.construction_activity_types = CHANGE_TYPES
        self.plot: bool = False
        self.translated_region: gpd.GeoDataFrame = None
        self.translate_sites: bool = True
        self.translated_sites: Dict = {}
        self.region_id_map_csvfile: os.PathLike = region_id_map_csvfile
        self._assess_region_id_mapping()

    def _assess_region_id_mapping(self):
        # load the csv as dictionary where dict[old_id] = new_id
        if (self.region_id_map_csvfile is not None) and os.path.exists(self.region_id_map_csvfile):
            self.region_id_map = pd.read_csv(region_id_map_csvfile, header=None, index_col=0, squeeze=True).to_dict()
        else:
            self.region_id_map = {}

    def find_files(self):
        json_paths = [
            os.path.join(self.input_directory, d)
            for d in natsorted(os.listdir(self.input_directory))
            if d.endswith(self.ext)
        ]
        return json_paths

    def _filter_data(self, data):
        return qfabric_activity_filter(data)

    def _translate_site(self) -> gpd.GeoDataFrame:
        return qfabric_to_smart_site_translator(self.site, self.activity_dates_dict, self.qsite_id, self.qregion_id)

    def translate_region(self):
        """Translates the region and its sites"""
        # single region
        if self.plot:
            plot_on_map(self.region, title=f"Qfabric Input Region: '{self.region_id}'", epsg=3857)
        self.translated_region, self.activity_dates_dict = qfabric_to_smart_region_translator(
            self.region, self.region_id, self.misc_region_info
        )
        if self.plot:
            plot_on_map(self.region, title=f"Qfabric Translated Region: '{self.region_id}'", epsg=3857)

        if self.translate_sites:
            # translate qfabric sites -> smart sites
            self.translated_sites = dict()
            # uses the region data immedaitely prior to being translated.
            site_labels = self.region.label.tolist()
            for i, site_label in enumerate(site_labels):
                qsite_id = f"{self.region_id}_{str(site_label).zfill(4)}"
                log.info(f"  > ({i + 1} of {len(site_labels)}); site_id: '{qsite_id}'")
                site = self.region[self.region["label"] == site_label]
                # site
                self.translated_sites[qsite_id] = qfabric_to_smart_site_translator(
                    site, self.activity_dates_dict, self.region_id, qsite_id
                )
                if self.plot:
                    plot_on_map(self.translated_site, title=f"{i}. Qfabric translated site: '{qsite_id}'", epsg=3857)

    def _translate_activity_labels(self, src_column: str):
        self.region[src_column] = self.region[src_column].map(self.activity_label_map)

    def translate_all(
        self,
        output_dir: os.PathLike,
        plot: bool = False,
        tag: str = None,
        translate_sites: bool = None,
        limit: Tuple = None,
    ):
        """Main.
        Iterates over all the QFabric annotation files.
        1. For each region
            a. try to load
            b. filter and parse
            c. translate region -> translated_region
            d. save translated_region (unique region_id -> <output_dir>/region_models/)
        2. if translate_sites == True:
            a.  parse site infor from region data
            b.  for each site:
                  i. translate site -> translated_site
                  ii. save translated_site (unique site_id -> <output_dir>/site_models/)
        """
        if tag is not None:
            self.tag = tag
        if translate_sites is not None:
            self.translate_sites = translate_sites
        if (limit is None) or (limit == "all"):
            start = 0
            end = self.ntotal_paths
        else:
            start = limit[0]
            end = limit[1]

        self.site_path = os.path.join(output_dir, "site_models")
        self.region_path = os.path.join(output_dir, "region_models")
        self.plot = plot

        # confirm directories:
        confirm_dir(directory=self.site_path)
        confirm_dir(directory=self.region_path)

        for index, filepath in enumerate(self.filepaths[start:end]):
            filename = os.path.basename(filepath)
            name = filename.split(".")[0]
            self.region_id = f"{self.tag}{str(name).zfill(3)}"

            # update region_id when region_id_map is available:
            if self.region_id in self.region_id_map:
                self.misc_region_info = f"{self.region_id}: {self.region_id_map[self.region_id]}"
                self.region_id = self.region_id_map[self.region_id]

            try:
                # load the region
                log.info(f"Loading the {index + start}-th region_id: '{self.region_id}':")
                self.region = gpd.read_file(filepath)
            except BaseException:
                self.failed_jsons.append(filepath)
                log.warning(f"**something went wrong with the '{filename}' file. Unable to load it.")
            finally:
                # filter region
                self.region = self._filter_data(self.region)
                log.debug(
                    f"\tregion_id: '{self.region_id}' "
                    f"\n\t q-region after filtering contains: {len(self.region)}-sites (i.e., rows)"
                )
                if len(self.region) > 0:
                    # translate region (and sites, iff translate_sites == True)
                    self.translate_region()
                    # save translated region
                    region_filepath = os.path.join(self.region_path, f"{self.region_id}.geojson")
                    self.translated_region.to_file(region_filepath, driver="GeoJSON")
                    self.translated_region_count += 1
                    for site_id, translated_site in self.translated_sites.items():
                        # save translated sites
                        site_filepath = os.path.join(self.site_path, f"{site_id}.geojson")
                        translated_site.to_file(site_filepath, driver="GeoJSON")
                        self.translated_site_count += 1
                    log.info(
                        f" >> {index}-th FILE. SUCCESFULLY PROCESSED "
                        f"QFABRIC: '{filename}' \n\t > as region_id: "
                        f"'{self.region_id}'"
                    )
                else:
                    log.info(
                        f" << {index}-th file ('{filename}') has no viable QFabric sites to process. "
                        f"\n\t region_id: '{self.region_id}' will not be created."
                    )
        print(
            f"COMPLETED THE TRANSLATION OF {self.translated_region_count}-regions"
            f" and {self.translated_site_count}-sites."
        )
        if len(self.failed_jsons) > 0:
            log.debug(
                f"\t >>> (there were {len(self.failed_jsons)} corrupted files, "
                f"which can be inspected via the '.failed_jsons' attribute)"
            )


if __name__ == "__main__":
    # the qfabric input/output locations
    WORK_PATH = "/data/qfabric"

    input_directory = os.path.join(WORK_PATH, "annotations/QFabric_Labels/geojsons/")
    region_id_map_csvfile = os.path.join("new-codes.csv")
    output_dir = os.path.join(WORK_PATH, "annotations/QFabric_Labels/smart_translated/v6/")

    input_directory = os.path.join(WORK_PATH, "geojsons")
    output_dir = os.path.join(WORK_PATH, "smart_translated/v6/")

    # initialize the translator
    qtranslator = QFABTranslator(
        input_directory=input_directory, ext=".geojson", region_id_map_csvfile=region_id_map_csvfile
    )
    print(f"DEFAULT TAG: {qtranslator.tag}")

    # run the translator:
    qtranslator.translate_all(
        output_dir=output_dir,  # where to save the annotations
        plot=False,  # plot=True, plots regions and sites using OpenStreetMaps
        tag="BLA_QFABRIC_R",  # tag used to append output filenames
        translate_sites=True,  # set to False to ommit sites (i.e., site_models/*.geojsons will not be created)
        limit=None,  # None or 'all': there is no limit / run all. Else, limit=[start: int, end: int]
    )

    # print the location of all the failed
    if len(qtranslator.failed_jsons) > 0:
        print(f"There are {len(qtranslator.failed_jsons)} files: ")
        for i, failed_file in enumerate(qtranslator.failed_jsons):
            print(f"  {i}. failed file: '{failed_file}'")
