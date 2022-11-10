from pathlib import Path
from natsort import natsorted
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import ogr
from osgeo import osr
import pyproj
import shapely.speedups
from shapely.geometry.base import BaseGeometry


CRS_4326 = pyproj.CRS("epsg:4326")
shapely.speedups.enable()


class SuperSites:
    def __init__(self, path: Path, crs: str = "EPSG:4326", n_sites: int = None, region_id: str = "BLA_001"):
        self.path: str = path
        self.crs: str = crs
        self.region_id = region_id
        self.data = gpd.GeoDataFrame()
        self.files: list = natsorted([x for x in Path(self.path).glob("**/*.geojson") if x.is_file()])
        self.all_sites_data = {}
        self.total_poly_count: int = 0
        if n_sites is not None:
            self.n_sites: int = n_sites
            self.files = self.files[:n_sites]
        self.version: str = "2.4.0"
        self.polytype: str = "superpolygon"
        # morph
        self.hulls: gpd.GeoDataFrame = None
        self.buffer: gpd.GeoDataFrame = None
        self.superpolys: gpd.GeoDataFrame = None
        self.buff_radius: float = 0.0
        # filtering
        self.accepted: gpd.GeoDataFrame = None
        self.rejected: gpd.GeoDataFrame = None
        self.filtered_sites: Dict = None
        self.criteria: Dict = None
        # load the sites data
        self._load()

    def _filter_data(self, criteria):
        self.criteria = criteria
        self.accepted = gpd.GeoDataFrame()
        self.rejected = gpd.GeoDataFrame()
        self.filtered_sites = {}
        for name, site_info in self.all_sites_data.items():
            temp = site_info["gdf"]
            # temp["name"] = name
            rejected = gpd.GeoDataFrame()
            if "score" in self.criteria:
                th = self.criteria["score"]
                temp, rejected = filter_scores(temp, th=th)
                self.filtered_sites[name] = temp
            else:
                self.criteria.update({"score": 0.0})

            if "pct_area" in self.criteria:
                th = self.criteria["pct_area"]
                temp, rejected = filter_areas(temp, th=th, how="pct_area")
                self.filtered_sites[name] = temp
            else:
                self.criteria.update({"pct_area": 0.0})

            if "min_area" in self.criteria:
                temp, rejected = filter_areas(temp, th=th, how="min_area")
                self.filtered_sites[name] = temp
            else:
                self.criteria.update({"min_area": 0.0})

            self.accepted = pd.concat([self.accepted, temp])
            self.rejected = pd.concat([self.rejected, rejected])

    def _load(self):
        if len(self.files) < 1:
            raise ValueError(f"Unable to find site_models files under the given location '{self.path}'")
        for file in self.files:
            name = file.name.split(".")[0]
            temp = gpd.read_file(file)
            temp = compute_area(temp)
            temp["name"] = name
            # all site data
            self.all_sites_data[name] = dict(gdf=temp)
            self.total_poly_count += len(temp)
            self.data = pd.concat([self.data, temp])
        # self.compute_area(temp)

    def _smooth_site_areas(self, criteria):
        """TODO: add a constraint to remove or replace site areas.
        E.g., if area_change (btwn sequential/short dates say 1-month) is > 25%,
        then replace by closest in time.
        integrate Andrew's method results.
        """
        pass

    def dilate_and_shrink(self, gdf: gpd.GeoDataFrame, buff_radius: float = 0.001, crs: str = None):
        if crs is None:
            crs = self.crs
        self.buff_radius = buff_radius
        # dilate
        buffer = add_buffer(gdf, radius=buff_radius)
        self.buffer = gpd.GeoDataFrame(buffer.unary_union, columns=["geometry"], crs=crs)
        # shrink
        shrink_radius = buff_radius * -1
        self.superpolys = add_buffer(self.buffer, radius=shrink_radius)

    def find_hulls(self):
        self.hulls = gpd.GeoDataFrame(self.data.unary_union, columns=["geometry"], crs=self.crs)

    def get_superpolys(self, buff_radius: float, criteria: Dict):
        # what data to use (i.e., strong scores)
        self._filter_data(criteria)
        # replace/update
        self.data = self.accepted
        # merge polygons
        self.find_hulls()
        self.dilate_and_shrink(self.hulls, buff_radius=buff_radius)

    def assign_observations_to_superpolys(self):
        """group the individual site observations into the identified/created
        super polys (SPOLY).
        """
        self.super_polys = {}
        for idx, superpoly in self.superpolys.iterrows():
            superpoly_name = f"{self.region_id}_{str(idx + 1).zfill(4)}"
            # print(f"{idx} super poly: {superpoly_name}")
            self.super_polys[superpoly_name] = gpd.GeoDataFrame()
            for _, site_info in self.all_sites_data.items():
                gdf = site_info["gdf"]
                intersections = gdf["geometry"].intersects(superpoly["geometry"])
                gdf["intersections"] = intersections
                gdf["spoly"] = superpoly_name
                n_intersections = np.sum(intersections.astype(int))
                if n_intersections > 0:
                    self.super_polys[superpoly_name] = pd.concat([self.super_polys[superpoly_name], gdf])

    def _summarize_site(self, superpoly_gdf: gpd.GeoDataFrame, superpoly_name: str) -> Dict:
        # columns for site summary
        current_phase = superpoly_gdf.current_phase.dropna()
        # more than one mode. select the first one.
        current_phase = current_phase.mode()[0] if len(current_phase) > 0 else None

        extra_cols = set(["area", "name", "intersections", "spoly"])
        # log_df = superpoly_gdf.copy()  # tracks morphological changes
        cols_to_drop = [c for c in extra_cols if c in superpoly_gdf.columns]
        superpoly_gdf = superpoly_gdf.drop(columns=cols_to_drop)
        mgrs = superpoly_gdf.mgrs.dropna().mode()
        mgrs = mgrs.tolist() if len(mgrs) > 0 else [None]

        site_summary = {
            "type": ["site"],
            "region_id": [self.region_id],
            "site_id": [superpoly_name],
            "version": [self.version],
            "mgrs": mgrs,
            "status": [superpoly_gdf.status.dropna().mode().item()],
            "start_date": [superpoly_gdf.start_date.dropna().min()],
            "end_date": [superpoly_gdf.end_date.dropna().max()],
            "model_content": [superpoly_gdf.model_content.dropna().mode().item()],
            "originator": [superpoly_gdf.originator.mode().item()],
            "score": [superpoly_gdf.score.mean()],
            "validated": [superpoly_gdf.validated.dropna().mode().item()],
            "observation_date": [superpoly_gdf.observation_date.dropna().min()],
            "source": [None],
            "sensor_name": [superpoly_gdf.sensor_name.dropna().mode().item()],
            "current_phase": [current_phase],
            "is_occluded": [None],
            "is_site_boundary": [None],
            "geometry": [superpoly_gdf.iloc[0].geometry],
            "misc_info": [
                dict(filtering_criteria=self.criteria, buffer_radius=self.buff_radius, polygon_type=self.polytype)
            ],
        }
        site_summary = gpd.GeoDataFrame.from_dict(site_summary, orient="columns", crs=self.crs)
        site_model = pd.concat([site_summary, superpoly_gdf[1:]])

        return dict(site_summary=site_summary, site_model=site_model)

    def get_site_summaries(self, buff_radius: float, criteria: Dict) -> Tuple[Dict, Dict]:
        self.criteria = criteria
        self.get_superpolys(buff_radius, self.criteria)  # across time
        self.assign_observations_to_superpolys()
        self.site_summaries = {}
        self.site_models = {}
        for superpoly_name, superpoly_gdf in self.super_polys.items():
            superpoly_data = self._summarize_site(superpoly_gdf, superpoly_name)
            self.site_models[superpoly_name] = superpoly_data["site_model"]
            self.site_summaries[superpoly_name] = superpoly_data["site_summary"]
        return self.site_models, self.site_summaries

    def site_models_to_geojson(self, outpath: Path, disp: bool = False):
        """outputpath: Path(f"{region_id}/site_models")"""
        sites_outpath = outpath / "site_models"
        confirm_directory(sites_outpath)
        for s, (site_name, site_model) in enumerate(self.site_models.items()):
            path = sites_outpath / f"{site_name}.geojson"
            site_model.to_file(path, driver="GeoJSON")
            if disp:
                print(f"{s}. name: {site_name} --> {path}")


class Region:
    """Takes in the prelimiary region_model and the site_summaries to produce a
    postprocessed region_model file with refined site summaries.
    """

    def __init__(self, path: Path, region_id: str, site_summaries: Dict):
        self.data: gpd.GeoDataFrame = gpd.read_file(path)
        self.region_id: str = region_id
        self.region: gpd.GeoDataFrame = self.data[:1]
        self.site_summaries: Dict = site_summaries
        self.region_model: gpd.GeoDataFrame = None

    def get_region_model(self, site_summaries=None):
        if site_summaries is not None:
            self.site_summaries = site_summaries
        # from site_summaries to geopandas
        sites_gdf = pd.concat(list(self.site_summaries.values())).reset_index(drop=True)
        region_cols = set(self.region.columns)
        site_cols = set(sites_gdf.columns)
        cols_to_drop = site_cols.difference(region_cols)
        sites_gdf.drop(columns=cols_to_drop, inplace=True)
        # additional adjustments
        sites_gdf["type"] = sites_gdf["type"].map({"site": "site_summary"})
        sites_gdf["region_id"] = None
        # combine region and assoc site_summary data
        self.region_model = pd.concat([self.region, sites_gdf]).reset_index(drop=True)
        return self.region_model

    def region_model_to_geojson(self, outpath: Path, disp: bool = False):
        """outputpath: Path(f"{region_id}/region_models")"""
        if self.region_model is None:
            self.get_region_model()
        confirm_directory(outpath / "region_models")
        path = outpath / f"region_models/{self.region_id}_BLA.geojson"
        self.region_model.to_file(path, driver="GeoJSON")
        if disp:
            print(f"Region model for id: {self.region_id} saved to '{path}'")


def filter_scores(gdf: gpd.GeoDataFrame, th: float = 0.0, target_column: str = "score") -> gpd.GeoDataFrame:
    if target_column in gdf:
        accept = gdf[gdf[target_column] >= th]
        reject = gdf.drop(index=accept.index)
    return accept, reject


def filter_areas(
    gdf: gpd.GeoDataFrame, th: float = 0.0, target_column: str = "area", how: str = None
) -> gpd.GeoDataFrame:
    max_area = gdf[target_column].max()

    if how == "pct_area":
        area_th = th * max_area
    elif how == "min_area":
        area_th = th
    else:
        raise NotImplementedError("Area filtering criteria is not currently implemented.")
        # return gdf, None

    if target_column in gdf:
        accept = gdf[gdf[target_column] >= area_th]
        reject = gdf.drop(index=accept.index)
    return accept, reject


def dicosmo_compute_area(gdf: gpd.GeoDataFrame, crs: str = "epsg:3857") -> gpd.GeoDataFrame:
    # TODO: this method still doesn work. error at "source.ImportFromEPSG(4326)"
    temp = gdf.to_crs(crs)
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    target = osr.SpatialReference()
    target.ImportFromEPSG(crs)
    transform = osr.CoordinateTransformation(source, target)
    poly = ogr.CreateGeometryFromJson(temp["geometry"])
    poly.Transform(transform)
    gdf["area"] = poly.GetArea()
    return gdf


def compute_area(
    gdf: gpd.GeoDataFrame, epsg: int = 3857, ratios: List[float] = [1, 10, 1e3], target_unit: int = 10
) -> gpd.GeoDataFrame:
    """Covenience method to compute and conver boundary areas.
    1: 1-meter units
    10: 10-meter units
    20: 20-meter units
    60: 60-meter units
    1e6: 1-kmeter units
    Scales:
        [m]^2 = 1m x 1m squares: 1E0
        [km]^2= 1km x 1km squares: 1E6 = 1E0 * 1E0* (1E3)^2 = 1[km]^2 * 1E6
        [10m]^2 = 10m x 10m squares: 1E4 = 1E1 * 1E1 * (1E2)^2 = [10m]^2 * 1E4
    Examples:
        Ex1: convert area units from [m]^2 to [km]^2
            conversion rate(rate): [m]^2 * 1e6 = [km]^2 -> [m]^2 = [km]^2 / 1e6
            Y = X / 1e6, X units in [m]^2 and Y units will be in [km]^2

        Ex2: convert area units from [m]^2 to [10m]^2
            rate: [m]^2 * 1e4 = [10m]^2 -> [m]^2 = [10m]^2 / 1e4
            Y = X / 1e4, X units in [m]^2 and Y units will be in [10m]^2

        Ex3: convert area units from [m]^2 to [20m]^2
            rate: [m]^2 * 2e4 = 2^2 * [10m]^2 -> [m]^2 = [10m]^2 / (20^2)
            Y = X / 1e4, X units in [m]^2 and Y units will be in [10m]^2
    """
    # project to planar space
    flat_geometry = gdf.to_crs(epsg=3857)
    # compute area in meters
    # area = flat_geometry["geometry"].area
    area_unit = flat_geometry.crs.axis_info[0].unit_name
    areas = gpd.GeoDataFrame()
    # convert area_unit to others [kilo-meter]^2, [10meter]^2
    for ratio in ratios:
        areas[f"area[{ratio}-{area_unit}]^2"] = gdf["geometry"].to_crs(epsg=epsg).map(lambda p: p.area / ratio**2)
    gdf["area"] = areas[f"area[{target_unit}-{area_unit}]^2"]
    return gdf


def buffer_meters_aeqd(geometry: BaseGeometry, meters, src_crs=CRS_4326, **kwargs) -> BaseGeometry:

    # find lat lon
    geom_crs_to_4236_transformer = pyproj.Transformer.from_crs(crs_from=src_crs, crs_to=CRS_4326, always_xy=True)
    center = shapely.ops.transform(geom_crs_to_4236_transformer.transform, geometry.centroid())
    lon, lat = center.x, center.y

    # Azimuthal equidistant projection https://proj.org/operations/projections/aeqd.html
    crs_aeqd = pyproj.CRS(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0")

    geom_crs_to_aeqd_transformer = pyproj.Transformer.from_crs(crs_from=src_crs, crs_to=crs_aeqd, always_xy=True)
    aeqd_geom = shapely.ops.transform(geom_crs_to_aeqd_transformer.transform, geometry)

    # buffer in meters (passing in additional arguments to control buffering process
    buffered_aeqd_geom = aeqd_geom.buffer(meters, **kwargs)

    # convert back to src_crs
    buffered_geom = shapely.ops.transform(geom_crs_to_aeqd_transformer.itransform, buffered_aeqd_geom)

    return buffered_geom


def add_buffer(gdf: gpd.GeoDataFrame, radius: float = 0.1, kwargs: dict = {"join_style": 2}) -> gpd.GeoDataFrame:
    """adds a radius buffer to a polygon
    see https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.buffer.html for more options.
    radius: float, buffer radius in range - to + (0<: shrinks; >0: expands).
    TODO: convert to meters, instead of latlon radius.
    """
    # gseries = gdf["geometry"].to_crs("epsg:4326")  # produces a warning
    gseries = gdf["geometry"].to_crs("epsg:3857")  # re-project geometries to a projected CRS
    temp = gpd.GeoDataFrame(gseries.buffer(radius, **kwargs), columns=["geometry"])
    gdf["geometry"] = temp["geometry"].to_crs("epsg:4326")
    return gdf


def confirm_directory(directory: os.PathLike):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_on_map(gdf: gpd.GeoDataFrame, title: str = "", figsize=(20, 20)):
    import contextily as ctx

    if "color" in gdf:
        colors = gdf["color"]
    else:
        colors = None
    if "alpha" in gdf:
        alpha = gdf["alpha"]
    else:
        alpha = 0.5

    ax = gdf.to_crs(epsg=3857).plot(figsize=figsize, alpha=alpha, edgecolor="k", facecolor=colors)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax


def run_postprocesing(
    input_dir: os.PathLike,
    region_id: str,
    out_dir: os.PathLike,
    filter_criteria: Dict = {"score": 0.0, "pct_area": 0.1, "min_area": 40},
    buff_radius: float = 0.05,
    disp: bool = False,
):
    """
    args:
        input_dir: path to the top level folder directory that contains the "debug" folder
            (contains the raw site_models and region_models folders).
        region_id: str, the name of the region to postprocess.
        filter_criteria: dictionary, this dictionary defined the filtering parameters
            applicable to score (default = {score: 0.6, area:0.1}).
            `score` filter removes polygons with score lower than the given float value
                (anything with score less than 0.6 is ignored).
            `pct_area` relative filter that removes polygons with area that is smaller than x-percent of
                the maximum area (i.e., relative thresholding).
            `min_area` threshold filter that removes polygons with area < `x: float` m^2.
            e.g., min_area: 40, 40m^2 min area.
        out_dir: path (default = None), an alternate location where the postprocessed
            site_models and region_models will be saved.
        buff_radius: float (default=0.002), degrees to erode and merge polygons in
            space; It then dilates the polygons to recover their (almost) original area,
            which slightly different for polygons that overlapped -- and were merged --
            after the erosion (no impact on the units or crs since erosion and dilation
            inverse operations). Example, buff_radius = 0.03 corresponds to 10 meters
    """
    # get data to postprocess from
    region_file = os.path.join(input_dir, "region_models", f"{region_id}_BLA.geojson")
    # sites to postprocess
    sites = SuperSites(path=Path(input_dir), region_id=region_id)
    _, _ = sites.get_site_summaries(buff_radius=buff_radius, criteria=filter_criteria)
    # save the post_processed site_model geojson files (one or more per region)
    sites.site_models_to_geojson(outpath=Path(out_dir), disp=disp)

    # region to postrprocess
    region = Region(region_file, region_id=region_id, site_summaries=sites.site_summaries)
    # save the post_processed region_model geojson file (one per region)
    region.region_model_to_geojson(outpath=Path(out_dir))
