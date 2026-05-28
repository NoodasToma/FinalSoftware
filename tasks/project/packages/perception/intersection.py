# One thing to flag: lane_mask in is_at_stop_line is assumed to be a binary/grayscale mask where nonzero pixels represent red stop-line pixels. If Person A's traffic light module or Person C's pipeline passes a different format (e.g. a BGR crop), you'll need to align on that before C5 lands.
from __future__ import annotations

import numpy as np

from tasks.project.packages.perception.apriltags import TagObservation
from tasks.project.packages.sign_registry import SignSemantic

_ALL_TURNS = {"left", "right", "straight"}
_CONSTRAINT_KINDS = {
    "4-way-intersect", "T-intersection", "right-T-intersect",
    "left-T-intersect", "oneway-right", "oneway-left", "do-not-enter",
}


def is_at_stop_line(tag_obs: TagObservation, lane_mask: np.ndarray) -> bool:
    if tag_obs.est_distance_m < 0.25:
        return True

    if tag_obs.side_length_px > 60:
        h = lane_mask.shape[0]
        bottom_third = lane_mask[2 * h // 3:, :]
        if int(np.count_nonzero(bottom_third)) > 500:
            return True

    return False


def merge_turn_constraints(observed_signs: list[SignSemantic]) -> set[str]:
    allowed = set(_ALL_TURNS)

    for sign in observed_signs:
        if sign.kind in _CONSTRAINT_KINDS:
            allowed &= sign.available_turns
        elif sign.kind == "no-left-turn":
            allowed.discard("left")
        elif sign.kind == "no-right-turn":
            allowed.discard("right")

    return allowed
