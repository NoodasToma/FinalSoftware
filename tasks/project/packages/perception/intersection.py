# One thing to flag: lane_mask in is_at_stop_line is assumed to be a binary/grayscale mask where nonzero pixels represent red stop-line pixels. If Person A's traffic light module or Person C's pipeline passes a different format (e.g. a BGR crop), you'll need to align on that before C5 lands.
from __future__ import annotations

import math

import numpy as np

from tasks.project.packages.perception.apriltags import TagObservation
from tasks.project.packages.sign_registry import SignSemantic

_ALL_TURNS = {"left", "right", "straight"}
_CONSTRAINT_KINDS = {
    "4-way-intersect", "T-intersection", "right-T-intersect",
    "left-T-intersect", "oneway-right", "oneway-left", "do-not-enter",
}


def is_at_stop_line(tag_obs: TagObservation, lane_mask: np.ndarray,
                    stop_distance_m: float = 0.25) -> bool:
    # Calibrated camera (real hardware AND the sim now that it has intrinsics):
    # stop when within stop_distance_m of the tag. This is THE path we want both
    # platforms to use. stop_distance_m is a calibration knob (maneuver_timings
    # `stop_distance_m`): it depends on how far the sign sits from the lane the
    # bot drives, so it is tuned per layout (sim) / per real Duckietown.
    if tag_obs.est_distance_m < stop_distance_m:
        return True

    # Pixel-size fallback ONLY when there is no metric distance at all, i.e. an
    # uncalibrated camera whose est_distance is inf. Without the isinf guard this
    # fired ~1 m early and pre-empted the real-distance path the instant
    # intrinsics WERE available (which made the sim stop/turn a metre too soon and
    # skip the hardware code path entirely). The proxy threshold is sized for the
    # sim's ~0.2 m tags so the bot stops while it can still see lane markings.
    if math.isinf(tag_obs.est_distance_m) and tag_obs.side_length_px > 38:
        h = lane_mask.shape[0]
        bottom_third = lane_mask[2 * h // 3:, :]
        if int(np.count_nonzero(bottom_third)) > 200:
            return True

    return False


_RED_LINE_DEFAULTS = {
    # Same red hue ranges as the traffic-light detector (red wraps the hue circle).
    "red_lower_h1": 0, "red_upper_h1": 10, "red_lower_h2": 170, "red_upper_h2": 180,
    "red_lower_s": 120, "red_upper_s": 255, "red_lower_v": 80, "red_upper_v": 255,
    # Bottom fraction of the frame to scan, and how many red pixels mean "line".
    "line_roi_frac": 0.22, "line_min_px": 400,
}


def detect_red_line(bgr_frame, cfg: dict | None = None) -> bool:
    """Is a red STOP LINE directly in front of the bot (i.e. are we AT the line)?

    Duckietown paints red stop lines across the lane at intersections; the
    hardware-correct stop trigger is seeing that line fill the bottom of the
    camera frame. With the camera pitched ~14 deg down, the bottom ~22% of rows
    only shows road within ~0.3 m of the bot, so this fires exactly when the
    front of the bot reaches the line — independent of sign-tag decode range.
    Red blobs elsewhere (octagon signs, traffic-light lenses) sit higher in the
    frame and never enter this ROI. Config keys (red_* shared with
    traffic_light_hsv.yaml, plus line_roi_frac / line_min_px) make it tunable
    for real-bot lighting.
    """
    import cv2

    if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
        return False
    c = dict(_RED_LINE_DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in _RED_LINE_DEFAULTS})

    h = bgr_frame.shape[0]
    roi = bgr_frame[int(h * (1.0 - float(c["line_roi_frac"]))):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo_s, hi_s = c["red_lower_s"], c["red_upper_s"]
    lo_v, hi_v = c["red_lower_v"], c["red_upper_v"]
    m1 = cv2.inRange(hsv, np.array([c["red_lower_h1"], lo_s, lo_v], np.uint8),
                     np.array([c["red_upper_h1"], hi_s, hi_v], np.uint8))
    m2 = cv2.inRange(hsv, np.array([c["red_lower_h2"], lo_s, lo_v], np.uint8),
                     np.array([c["red_upper_h2"], hi_s, hi_v], np.uint8))
    return int(cv2.countNonZero(m1) + cv2.countNonZero(m2)) >= int(c["line_min_px"])


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
