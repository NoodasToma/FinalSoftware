from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "traffic_light_hsv.yaml"
)

# Fallback bounds, used when the YAML is missing or a key is absent, so the
# detector never crashes on a partial config. Mirrors traffic_light_hsv.yaml.
_DEFAULT_HSV = {
    "red_lower_h1": 0,    "red_upper_h1": 10,
    "red_lower_h2": 170,  "red_upper_h2": 180,
    "red_lower_s": 120,   "red_upper_s": 255,
    "red_lower_v": 80,    "red_upper_v": 255,
    "yellow_lower_h": 20, "yellow_upper_h": 35,
    "yellow_lower_s": 100, "yellow_upper_s": 255,
    "yellow_lower_v": 100, "yellow_upper_v": 255,
    "green_lower_h": 40,  "green_upper_h": 85,
    "green_lower_s": 80,  "green_upper_s": 255,
    "green_lower_v": 60,  "green_upper_v": 255,
    "min_area": 80,
}

# BGR colours used to draw each detected blob in the debug overlay.
_DRAW_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}


def _bound(h, s, v) -> np.ndarray:
    return np.array([h, s, v], dtype=np.uint8)


class TrafficLightDetector:
    """Turns a BGR camera frame into 'red' | 'yellow' | 'green' | None.

    Detection only runs while armed (Person C arms it when a 't-light-ahead'
    AprilTag is seen and disarms it once past the intersection).
    """

    def __init__(self, hsv_config_path: Optional[str] = None) -> None:
        path = hsv_config_path or _DEFAULT_CONFIG_PATH
        cfg = dict(_DEFAULT_HSV)
        try:
            with open(path) as fh:
                loaded = yaml.safe_load(fh) or {}
            cfg.update(loaded)
        except FileNotFoundError:
            pass

        self._min_area = float(cfg["min_area"])
        self._kernel = np.ones((3, 3), np.uint8)
        # Per-colour HSV ranges; red spans two ranges to cover the hue wrap.
        self._ranges = {
            "red": [
                (_bound(cfg["red_lower_h1"], cfg["red_lower_s"], cfg["red_lower_v"]),
                 _bound(cfg["red_upper_h1"], cfg["red_upper_s"], cfg["red_upper_v"])),
                (_bound(cfg["red_lower_h2"], cfg["red_lower_s"], cfg["red_lower_v"]),
                 _bound(cfg["red_upper_h2"], cfg["red_upper_s"], cfg["red_upper_v"])),
            ],
            "yellow": [
                (_bound(cfg["yellow_lower_h"], cfg["yellow_lower_s"], cfg["yellow_lower_v"]),
                 _bound(cfg["yellow_upper_h"], cfg["yellow_upper_s"], cfg["yellow_upper_v"])),
            ],
            "green": [
                (_bound(cfg["green_lower_h"], cfg["green_lower_s"], cfg["green_lower_v"]),
                 _bound(cfg["green_upper_h"], cfg["green_upper_s"], cfg["green_upper_v"])),
            ],
        }
        self._armed = False

    def arm(self) -> None:
        """Turn detection on."""
        self._armed = True

    def disarm(self) -> None:
        """Turn detection off (saves CPU); detect() then always returns None."""
        self._armed = False

    def _color_mask(self, hsv, ranges) -> np.ndarray:
        """Union the (lower, upper) ranges for a colour, then morph-open (3x3)."""
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def _qualifying_blobs(self, mask):
        """Contours of `mask` passing the area + square-ish aspect check.

        Returns [(area, (x, y, w, h)), ...]; bbox coords are relative to the
        mask the caller built (the top-half ROI here).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blobs = []
        for c in contours:
            area = float(cv2.contourArea(c))
            x, y, w, bh = cv2.boundingRect(c)
            aspect = (w / float(bh)) if bh > 0 else 0.0
            if area >= self._min_area and 0.5 <= aspect <= 2.0:
                blobs.append((area, (x, y, w, bh)))
        return blobs

    def _largest_blob_area(self, mask) -> float:
        """Area of the largest contour passing the area + aspect check, or 0.0."""
        return max((area for area, _ in self._qualifying_blobs(mask)), default=0.0)

    def detect(self, bgr_frame) -> Optional[str]:
        """Returns 'red' | 'yellow' | 'green' | None.

        Always returns None when not armed. Considers all three colours and
        returns the one whose largest qualifying blob is biggest. Lights sit
        above the road, so the search is restricted to the top half of the frame.
        """
        if not self._armed:
            return None
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            return None

        h = bgr_frame.shape[0]
        hsv = cv2.cvtColor(bgr_frame[: h // 2], cv2.COLOR_BGR2HSV)

        best_color: Optional[str] = None
        best_area = 0.0
        for color, ranges in self._ranges.items():
            area = self._largest_blob_area(self._color_mask(hsv, ranges))
            if area > best_area:
                best_area = area
                best_color = color
        return best_color

    def get_debug_overlay(self, bgr_frame) -> np.ndarray:
        """Returns a copy of bgr_frame annotated for the video stream.

        Each qualifying blob is drawn as a rectangle in its own colour, and the
        result of detect() ('red'|'yellow'|'green'|'none') is stamped top-left.
        When disarmed, stamps 'DISARMED' top-left and returns the frame
        otherwise unmodified.
        """
        out = bgr_frame.copy()
        if not self._armed:
            cv2.putText(out, "DISARMED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return out

        h = bgr_frame.shape[0]
        hsv = cv2.cvtColor(bgr_frame[: h // 2], cv2.COLOR_BGR2HSV)
        for color, ranges in self._ranges.items():
            mask = self._color_mask(hsv, ranges)
            for _area, (x, y, w, bh) in self._qualifying_blobs(mask):
                # ROI is the top half (y origin 0), so bbox coords map directly.
                cv2.rectangle(out, (x, y), (x + w, y + bh), _DRAW_BGR[color], 2)

        label = self.detect(bgr_frame) or "none"
        cv2.putText(out, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return out


def should_brake_for_yellow(color, yellow_started_at, now) -> bool:
    """Conservative yellow-light brake decision.

    Returns True iff the light is yellow AND has been yellow for more than 0.5s
    (so we brake for a settled yellow rather than diving through a fresh one).
    Returns False in every other case, including color None/'red'/'green' or
    yellow_started_at is None.

    Examples:
        should_brake_for_yellow('yellow', 10.0, 10.6) -> True
        should_brake_for_yellow('yellow', 10.0, 10.3) -> False
        should_brake_for_yellow('red',    10.0, 99.0) -> False
        should_brake_for_yellow(None,     10.0, 99.0) -> False
    """
    return (
        color == "yellow"
        and yellow_started_at is not None
        and (now - yellow_started_at) > 0.5
    )
