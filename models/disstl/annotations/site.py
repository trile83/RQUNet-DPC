""" site.py defines Site, which stores site-specific metadata """
import bisect
from datetime import date
from typing import Dict, List, Optional, Union

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from disstl.annotations.enums import Activity, SiteStatus
from disstl.annotations.site_observation import SiteObservation
from disstl.annotations.utils import as_date, as_date_str


class Site:
    """Site stores site-specific metadata"""

    # Value used for unlabeled site_masks
    UNLABELED = 0

    def __init__(  # TODO determine which fields are not optional.
        self,
        geometry: BaseGeometry,  # lat/long coordinates
        region_id: Optional[str] = None,
        site_id: Optional[str] = None,
        version: Optional[str] = None,
        mgrs: Optional[str] = None,
        status: Optional[Union[SiteStatus, str]] = SiteStatus.SYSTEM_CONFIRMED,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
        model_content: Optional[str] = "proposed",
        originator: Optional[str] = "bla",
        score: Optional[float] = None,
        validated: Optional[Union[bool, str]] = False,
        predicted_phase_transition: Optional[Union[Activity, str]] = None,
        predicted_phase_transition_date: Optional[Union[date, str]] = None,
        misc_info: Optional[dict] = None,
        observations: Optional[List[SiteObservation]] = None,
    ):
        self.geometry = geometry
        self.region_id = region_id
        self.site_id = site_id
        self.site_id_int = int(site_id.split("_")[-1]) + 1  # "a_site_id_0000" becomes 1. Used in site masks
        self.version = version
        self.mgrs = mgrs
        self.status = status
        if type(status) is str:
            self.status = SiteStatus.from_text(status)
        self.start_date = as_date(start_date)
        self.end_date = as_date(end_date)
        self.model_content = model_content
        self.originator = originator
        self.score = score
        self.validated = str(validated) == "True"
        self.predicted_phase_transition = predicted_phase_transition
        if type(predicted_phase_transition) is str:
            self.predicted_phase_transition = Activity.from_text(predicted_phase_transition)
        self.predicted_phase_transition_date = as_date(predicted_phase_transition_date)
        self.misc_info = misc_info
        if self.misc_info is None:
            self.misc_info = {}
        self._observation_dates = []  # helper list for maintaining a sorted list of observation by date
        self.observations = []
        if observations is not None:
            self.observations = sorted(observations, key=lambda obs: obs.observation_date)
            self._observation_dates = list(map(lambda obs: obs.observation_date, self.observations))

    def __eq__(self, other):
        return (type(other) is type(self)) and (self.__dict__ == other.__dict__)

    def _add_observation(self, observation):
        if not observation.observation_date:
            return
        if self.status == SiteStatus.NEGATIVE:
            observation.current_phase = (Activity.NO_ACTIVITY,)
        if self.observations:
            insertion_index = bisect.bisect(self._observation_dates, observation.observation_date)
        else:
            insertion_index = 0
        self.observations.insert(insertion_index, observation)
        self._observation_dates.insert(insertion_index, observation.observation_date)

    @classmethod
    def from_region_model_geojson_item(cls, region_geojson_item: Dict):
        """returns a Site from the given region model GeoJson item (site-summary)"""
        props = region_geojson_item["properties"]
        return cls(
            shape(region_geojson_item["geometry"]),
            site_id=props.get("site_id"),
            version=props.get("version"),
            mgrs=props.get("mgrs"),
            status=props.get("status"),
            start_date=props.get("start_date"),
            end_date=props.get("end_date"),
            model_content=props.get("model_content"),
            originator=props.get("originator"),
            score=props.get("score"),
            validated=props.get("validated"),
        )

    @classmethod
    def from_geojson(cls, site_geojson: Dict):
        """returns a Site from the given site model geojson"""
        site_item = site_geojson["features"][0]
        props = site_item["properties"]
        site = cls(
            shape(site_item["geometry"]),
            region_id=props.get("region_id"),
            site_id=props.get("site_id"),
            version=props.get("version"),
            mgrs=props.get("mgrs"),
            status=props.get("status"),
            start_date=props.get("start_date"),
            end_date=props.get("end_date"),
            model_content=props.get("model_content"),
            originator=props.get("originator"),
            score=props.get("score"),
            validated=props.get("validated"),
            predicted_phase_transition=props.get("predicted_phase_transition"),
            predicted_phase_transition_date=props.get("predicted_phase_transition_date"),
            misc_info=props.get("misc_info"),
        )
        for observation_item in site_geojson["features"][1:]:
            site._add_observation(SiteObservation.from_site_geojson_item(observation_item))
        return site

    def to_geojson(self):
        return {
            "type": "FeatureCollection",
            "features": [self.to_site_model_geojson_item()]
            + [observation.to_site_model_geojson_item() for observation in self.observations],
        }

    def to_region_model_geojson_item(self):
        return {
            "type": "Feature",
            "properties": {
                "type": "site_summary",
                "site_id": self.site_id,
                "version": self.version,
                "mgrs": self.mgrs,
                "status": SiteStatus.to_text(self.status),
                "start_date": as_date_str(self.start_date),
                "end_date": as_date_str(self.end_date),
                "model_content": self.model_content,
                "originator": self.originator,
                "score": self.score,
                "validated": str(self.validated),
            },
            "geometry": mapping(self.geometry),
        }

    def to_site_model_geojson_item(self):
        output = {
            "type": "Feature",
            "properties": {
                "type": "site",
                "region_id": self.region_id,
                "site_id": self.site_id,
                "version": self.version,
                "mgrs": self.mgrs,
                "status": SiteStatus.to_text(self.status),
                "start_date": as_date_str(self.start_date),
                "end_date": as_date_str(self.end_date),
                "model_content": self.model_content,
                "originator": self.originator,
                "score": self.score,
                "validated": str(self.validated),
                "misc_info": self.misc_info,
            },
            "geometry": mapping(self.geometry),
        }
        if self.predicted_phase_transition is not None:
            output["properties"]["predicted_phase_transition"] = Activity.to_text(
                *self.predicted_phase_transition, set_unlabeled_to_unknown=True
            )
            output["properties"]["predicted_phase_transition_date"] = as_date_str(self.predicted_phase_transition_date)
        return output

    def get_observation_for_date(self, date: date):
        if not self.observations or (
            (self.start_date is not None and date < self.start_date)
            or (self.end_date is not None and date > self.end_date)
        ):
            return None
        off_by_one = bisect.bisect(self._observation_dates, date)
        if off_by_one == 0:
            observation_index = 0
        else:
            observation_index = off_by_one - 1
        return self.observations[observation_index]

    def get_observation_with_observation_date(self, date: date) -> Optional[SiteObservation]:
        """
        Returns SiteObservation, only if it exists on date
        :param date: the date to search with
        :return: a SiteObservation if it exists on that date, else None
        """
        if not self.observations:
            return None
        return next((o for o in self.observations if o.observation_date == date), None)
