import json
import mgrs
import pyproj

import shapely.ops
import shapely.wkt
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.warp import transform_geom
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

EPSG_4326 = pyproj.CRS("EPSG:4326")
MGRS = mgrs.MGRS()


def get_utm_crs_for_latlon(latitude: float, longitude: float) -> pyproj.CRS:
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude, south_lat_degree=latitude, east_lon_degree=longitude, north_lat_degree=latitude
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def buffer_meters(geometry: BaseGeometry, meters: float, src_crs=EPSG_4326, **kwargs):
    point = geometry.representative_point()
    crs_utm = get_utm_crs_for_latlon(point.y, point.x)
    to_utm_meters = pyproj.Transformer.from_crs(src_crs, crs_utm, always_xy=True)
    geometry_in_utm = shapely.ops.transform(to_utm_meters.transform, geometry)
    buffered = geometry_in_utm.buffer(meters, **kwargs)
    return shapely.ops.transform(to_utm_meters.itransform, buffered)


def load_geometry(geometry_string: str) -> BaseGeometry:
    """
    Parses a string in either GeoJSON or WKT format into shapely object
    :param geometry_string:
    :return: shapely object
    """
    print(f"parsing geometry {geometry_string}")
    try:
        site_geometry = wkt.loads(geometry_string)
    except Exception:
        try:
            site_geometry = shape(json.loads(geometry_string))
        except Exception:
            print("geometry must be GeoJSON or WKT format.")
            raise

    return site_geometry


def from_4326(geometry: BaseGeometry, dst_crs: pyproj.CRS) -> BaseGeometry:
    return shape(transform_geom(EPSG_4326, dst_crs, geometry))
