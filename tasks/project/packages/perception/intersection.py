# One thing to flag: lane_mask in is_at_stop_line is assumed to be a binary/grayscale mask where nonzero pixels represent red stop-line pixels. If Person A's traffic light module or Person C's pipeline passes a different format (e.g. a BGR crop), you'll need to align on that before C5 lands.
import math
from typing import List, Optional, Set

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


def red_line_threshold(cfg: Optional[dict] = None) -> int:
    """The red-pixel count in the bottom ROI that means 'AT the line'
    (line_min_px), after applying any per-environment override in cfg."""
    c = dict(_RED_LINE_DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in _RED_LINE_DEFAULTS})
    return int(c["line_min_px"])


def red_line_pixel_count(bgr_frame, cfg: Optional[dict] = None) -> int:
    """How many red pixels are in the bottom-of-frame stop-line ROI.

    Split out from detect_red_line so callers can both decide (count >=
    threshold) AND surface the raw count for live tuning — on an uncalibrated
    bot the red line is the PRIMARY stop trigger, so seeing the actual count vs
    line_min_px is how you tell 'stopped short of the line' (count never reached
    threshold) from a false stop (count spikes on non-line red) and tune the
    red_* / line_min_px knobs for the venue. Returns 0 for an empty frame."""
    import cv2

    if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
        return 0
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
    return int(cv2.countNonZero(m1) + cv2.countNonZero(m2))


def detect_red_line(bgr_frame, cfg: Optional[dict] = None) -> bool:
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
    return red_line_pixel_count(bgr_frame, cfg) >= red_line_threshold(cfg)


_RED_BAND_DEFAULTS = {
    # TIGHTER red than the raw-count detector: the venue's orange road markings
    # leak into the wide red range and cause the count to spike far from any line,
    # so the band test uses a narrow, saturated red.
    "red_lower_h1": 0, "red_upper_h1": 8, "red_lower_h2": 172, "red_upper_h2": 180,
    "red_lower_s": 150, "red_upper_s": 255, "red_lower_v": 80, "red_upper_v": 255,
    "band_roi_frac": 0.32,      # bottom fraction of the frame to scan
    "band_center_frac": 0.70,   # only the central X of the width (the lane ahead, not side markings)
    "band_row_min_frac": 0.50,  # a ROW counts as "on the bar" if >= this fraction of the centre window is red
    "band_min_rows_frac": 0.14, # >= this fraction of ROI rows must be "on the bar" => a real horizontal BAND
    "band_max_rows_frac": 0.92, # ... but not essentially the WHOLE ROI (that's a red floor /全-red noise)
}


def red_band_metrics(bgr_frame, cfg: Optional[dict] = None):
    """Detect the painted RED STOP LINE as a wide HORIZONTAL BAND (not a raw pixel
    count). In a venue full of red/orange markings the plain pixel count spikes
    far from any line; a real stop line is different in STRUCTURE: across the
    bottom-centre of the frame, many CONSECUTIVE-ish rows are red across most of
    the lane width. Scattered dashes / diagonal hatching never fill rows that way.

    Returns (present: bool, diag: dict) where diag has:
      rows       — number of rows in the ROI that are 'on the bar'
      rows_frac  — rows / ROI_row_count
      row_max    — the single widest row's red coverage (fraction of the window)
    row_max is the key tuning signal: it only approaches ~1.0 when red actually
    spans the lane (i.e. the bot is AT the line), and stays low over orange noise.
    """
    import cv2

    c = dict(_RED_BAND_DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in _RED_BAND_DEFAULTS})
    diag = {"rows": 0, "rows_frac": 0.0, "row_max": 0.0}
    if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
        return False, diag

    h, w = bgr_frame.shape[:2]
    y0 = int(h * (1.0 - float(c["band_roi_frac"])))
    cw = float(c["band_center_frac"])
    x0 = int(w * (1.0 - cw) / 2.0)
    x1 = int(w * (1.0 + cw) / 2.0)
    roi = bgr_frame[y0:, x0:x1]
    if roi.size == 0:
        return False, diag
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_lo, s_hi = c["red_lower_s"], c["red_upper_s"]
    v_lo, v_hi = c["red_lower_v"], c["red_upper_v"]
    m1 = cv2.inRange(hsv, np.array([c["red_lower_h1"], s_lo, v_lo], np.uint8),
                     np.array([c["red_upper_h1"], s_hi, v_hi], np.uint8))
    m2 = cv2.inRange(hsv, np.array([c["red_lower_h2"], s_lo, v_lo], np.uint8),
                     np.array([c["red_upper_h2"], s_hi, v_hi], np.uint8))
    mask = m1 | m2
    win_w = mask.shape[1]
    if win_w == 0 or mask.shape[0] == 0:
        return False, diag
    row_cov = mask.sum(axis=1) / (255.0 * win_w)     # per-row red coverage [0..1]
    on_bar = int(np.count_nonzero(row_cov >= float(c["band_row_min_frac"])))
    rows_frac = on_bar / float(mask.shape[0])
    diag = {"rows": on_bar, "rows_frac": round(rows_frac, 3),
            "row_max": round(float(row_cov.max()), 3)}
    present = (float(c["band_min_rows_frac"]) <= rows_frac <= float(c["band_max_rows_frac"]))
    return present, diag


def merge_turn_constraints(observed_signs: List[SignSemantic]) -> Set[str]:
    allowed = set(_ALL_TURNS)

    for sign in observed_signs:
        if sign.kind in _CONSTRAINT_KINDS:
            allowed &= sign.available_turns
        elif sign.kind == "no-left-turn":
            allowed.discard("left")
        elif sign.kind == "no-right-turn":
            allowed.discard("right")

    return allowed
