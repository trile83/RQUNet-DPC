""" utils.py defines utility methods for use in the annotations module """
from datetime import date, datetime
from typing import Union
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
import shapely.geometry.polygon

from disstl.standards import STD_DATE_FORMAT


def as_date(str_or_date: Union[str, date]) -> date:
    return datetime.strptime(str_or_date, STD_DATE_FORMAT).date() if type(str_or_date) is str else str_or_date


def as_date_str(date: date) -> str:
    return date.strftime(STD_DATE_FORMAT) if date is not None else None


def orient_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.geom_type == "Polygon":
        geometry = MultiPolygon([shapely.geometry.polygon.orient(geometry)])
    else:
        geometry = MultiPolygon([shapely.geometry.polygon.orient(c) for c in geometry.geoms])

    return geometry
