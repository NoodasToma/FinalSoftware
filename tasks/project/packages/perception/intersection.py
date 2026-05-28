# One thing to flag: lane_mask in is_at_stop_line is assumed to be a binary/grayscale mask where nonzero pixels represent red stop-line pixels. If Person A's traffic light module or Person C's pipeline passes a different format (e.g. a BGR crop), you'll need to align on that before C5 lands.

from __future__ import annotations

import numpy as np

from tasks.project.packages.perception.apriltags import TagObservation
from tasks.project.packages.sign_registry import SignSemantic

_ALL_TURNS = {"left", "right", "straight"}
_EXCLUSION_SIGNS = {"no-left-turn", "no-right-turn", "do-not-enter"}


def is_at_stop_line(tag_obs: TagObservation, lane_mask: np.ndarray) -> bool:
    if tag_obs.est_distance_m < 0.25:
        return True

    if tag_obs.side_length_px > 60:
        h = lane_mask.shape[0]
        bottom_third = lane_mask[int(h * 2 / 3):, :]
        red_pixel_count = int(np.count_nonzero(bottom_third))
        threshold = bottom_third.size * 0.05
        if red_pixel_count > threshold:
            return True

    return False


def merge_turn_constraints(observed_signs: list[SignSemantic]) -> set[str]:
    allowed = set(_ALL_TURNS)

    for sign in observed_signs:
        if sign.kind == "no-left-turn":
            allowed.discard("left")
        elif sign.kind == "no-right-turn":
            allowed.discard("right")
        elif sign.kind == "do-not-enter":
            allowed.clear()
        else:
            allowed &= sign.available_turns

    return allowed

