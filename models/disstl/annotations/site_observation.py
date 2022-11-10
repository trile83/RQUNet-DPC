from datetime import date
from typing import Dict, Optional, Tuple, Union

from shapely.geometry import mapping, shape, MultiPolygon
from shapely.geometry.base import BaseGeometry

from disstl.annotations.enums import Activity
from disstl.annotations.utils import as_date, as_date_str

Window = Tuple[Tuple[int, int], Tuple[int, int]]


def as_bool_tuple(bool_or_str):
    if type(bool_or_str) is str:
        return tuple([item == "True" for item in bool_or_str.split(", ")])
    return bool_or_str


def bools_as_str(bools):
    if bools is not None:
        return ", ".join([str(b) for b in bools])
    return None


class SiteObservation:
    """SiteObservation contains the geometry and metadata of a site over a given time period"""

    def __init__(  # TODO determine which fields are not optional.
        self,
        geometry: BaseGeometry,
        current_phase: Union[Tuple[Activity], str],
        window: Optional[Window] = None,  # pixel coordinates
        observation_date: Optional[Union[date, str]] = None,
        source: Optional[str] = None,
        sensor_name: Optional[str] = None,
        is_occluded: Optional[Union[Tuple[bool], str]] = None,
        is_site_boundary: Optional[Union[Tuple[bool], str]] = None,
        score: Optional[float] = None,
        misc_info: Optional[dict] = None,
    ):
        self.observation_date = as_date(observation_date)
        self.source = source
        self.sensor_name = sensor_name
        self.current_phase = current_phase or (Activity.UNKNOWN,)
        if type(current_phase) is str:
            self.current_phase = Activity.from_text(current_phase)
        self.is_occluded = as_bool_tuple(is_occluded or (False,))
        self.is_site_boundary = as_bool_tuple(is_site_boundary or (True,))
        self.score = score
        self.misc_info = misc_info
        self.geometry = MultiPolygon([])
        if not isinstance(geometry, MultiPolygon):
            self.geometry = MultiPolygon([geometry])
        else:
            self.geometry = geometry
        self.window = window

    def __eq__(self, other):
        return (type(other) is type(self)) and (self.__dict__ == other.__dict__)

    @classmethod
    def from_site_geojson_item(cls, snapshot_geojson: Dict):
        props = snapshot_geojson["properties"]
        return cls(
            shape(snapshot_geojson["geometry"]),
            observation_date=props.get("observation_date"),
            source=props.get("source"),
            sensor_name=props.get("sensor_name"),
            current_phase=props.get("current_phase"),
            is_occluded=props.get("is_occluded"),
            is_site_boundary=props.get("is_site_boundary"),
            score=props.get("score"),
            misc_info=props.get("misc_info"),
        )

    def to_site_model_geojson_item(self):
        current_phase = self.current_phase
        if self.current_phase is None:
            current_phase = (Activity.UNKNOWN,)
        return {
            "type": "Feature",
            "properties": {
                "type": "observation",
                "observation_date": as_date_str(self.observation_date),
                "source": self.source,
                "sensor_name": self.sensor_name,
                "current_phase": Activity.to_text(*current_phase, set_unlabeled_to_unknown=True),
                "is_occluded": bools_as_str(self.is_occluded),
                "is_site_boundary": bools_as_str(self.is_site_boundary),
                "score": self.score,
                "misc_info": self.misc_info,
            },
            "geometry": mapping(self.geometry),
        }
