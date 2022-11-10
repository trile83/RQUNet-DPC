from __future__ import annotations

from enum import IntEnum
from typing import Dict

from bidict import bidict


class Activity(IntEnum):

    UNLABELED = 0
    SITE_PREPARATION = 1
    ACTIVE_CONSTRUCTION = 2
    POST_CONSTRUCTION = 3
    UNKNOWN = 4
    NO_ACTIVITY = 5

    @staticmethod
    def from_text(text: str):
        activities = []
        for key in text.split(","):
            key = key.strip()
            if key not in _activity_from_text:
                raise ValueError(f"{key} is not a recognized class of construction activity.")
            activities.append(Activity(_activity_from_text[key]))
        return tuple(sorted(activities))

    @staticmethod
    def to_text(*activities, set_unlabeled_to_unknown=False) -> str:
        # use a list here instead of a set to guarantee the order of the output
        activity_classes = []
        for activity in sorted(activities):
            if set_unlabeled_to_unknown and activity == Activity.UNLABELED:
                text = _activity_from_text.inverse[Activity.UNKNOWN]
            else:
                text = _activity_from_text.inverse[activity]
            activity_classes.append(text)
        return ", ".join(list(activity_classes))

    @staticmethod
    def names_by_id() -> Dict[Activity, str]:
        return dict(_activity_from_text.inverse)


_activity_from_text = bidict(
    {
        "Unlabeled": Activity.UNLABELED,
        "Site Preparation": Activity.SITE_PREPARATION,
        "Active Construction": Activity.ACTIVE_CONSTRUCTION,
        "Post Construction": Activity.POST_CONSTRUCTION,
        "Unknown": Activity.UNKNOWN,
        "No Activity": Activity.NO_ACTIVITY,
    }
)


class SiteStatus(IntEnum):

    POSITIVE_ANNOTATED = 0
    POSITIVE_PARTIAL = 1
    POSITIVE_ANNOTATED_STATIC = 2
    POSITIVE_PARTIAL_STATIC = 3
    POSITIVE_PENDING = 4
    POSITIVE_EXCLUDED = 5
    NEGATIVE = 6
    IGNORE = 7
    SYSTEM_PROPOSED = 8
    SYSTEM_CONFIRMED = 9
    SYSTEM_REJECTED = 10

    @staticmethod
    def from_text(text: str):
        key = text.strip()
        if key not in _site_status_from_text:
            raise ValueError(f"{key} is not a recognized site status.")
        return _site_status_from_text[key]

    def to_text(status: int):
        if status not in _site_status_from_text.inverse:
            raise ValueError(f"{status} is not a recognized site status enum.")
        return _site_status_from_text.inverse[status]


_site_status_from_text = bidict(
    {
        "positive_annotated": SiteStatus.POSITIVE_ANNOTATED,
        "positive_partial": SiteStatus.POSITIVE_PARTIAL,
        "positive_annotated_static": SiteStatus.POSITIVE_ANNOTATED_STATIC,
        "positive_partial_static": SiteStatus.POSITIVE_PARTIAL_STATIC,
        "positive_pending": SiteStatus.POSITIVE_PENDING,
        "positive_excluded": SiteStatus.POSITIVE_EXCLUDED,
        "negative": SiteStatus.NEGATIVE,
        "ignore": SiteStatus.IGNORE,
        "system_proposed": SiteStatus.SYSTEM_PROPOSED,
        "system_confirmed": SiteStatus.SYSTEM_CONFIRMED,
        "system_rejected": SiteStatus.SYSTEM_REJECTED,
    }
)
