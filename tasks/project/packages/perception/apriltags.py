from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_CAMERA_CONFIG_SEARCH_PATHS = [
    # Relative to repo root when running on the bot
    "duckiebot/camera_driver/config/camera_config.yaml",
    # Alternate location seen on some Duckiebot images
    "/data/config/calibrations/camera_intrinsic/default.yaml",
]

_DEFAULT_INTRINSICS = dict(fx=320.0, fy=320.0, cx=320.0, cy=240.0)


def _load_intrinsics() -> tuple[float, float, float, float]:
    """
    Try to read fx, fy, cx, cy from a camera-config YAML file.

    Returns (fx, fy, cx, cy).  Falls back to Duckietown defaults and logs a
    warning **once** when the fallback is used.
    """
    # yaml is a soft dependency & only needed here
    try:
        import yaml  # PyYAML ships with the Duckiebot image
    except ImportError:
        logger.warning(
            "apriltags: PyYAML not found — using default camera intrinsics "
            "(fx=fy=320, cx=320, cy=240).  Distance estimates may be inaccurate."
        )
        return _DEFAULT_INTRINSICS["fx"], _DEFAULT_INTRINSICS["fy"], \
               _DEFAULT_INTRINSICS["cx"], _DEFAULT_INTRINSICS["cy"]

    for path in _CAMERA_CONFIG_SEARCH_PATHS:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
            # Accept both flat keys and nested under 'camera_matrix'
            if all(k in cfg for k in ("fx", "fy", "cx", "cy")):
                fx, fy = float(cfg["fx"]), float(cfg["fy"])
                cx, cy = float(cfg["cx"]), float(cfg["cy"])
                logger.info("apriltags: loaded camera intrinsics from %s", path)
                return fx, fy, cx, cy
            # ROS-style: camera_matrix.data = [fx,0,cx, 0,fy,cy, 0,0,1]
            if "camera_matrix" in cfg and "data" in cfg["camera_matrix"]:
                d = cfg["camera_matrix"]["data"]
                fx, cx = float(d[0]), float(d[2])
                fy, cy = float(d[4]), float(d[5])
                logger.info(
                    "apriltags: loaded camera_matrix intrinsics from %s", path
                )
                return fx, fy, cx, cy
        except Exception as exc:  
            logger.debug("apriltags: could not parse %s — %s", path, exc)

    logger.warning(
        "apriltags: camera intrinsics not found in any of %s — "
        "falling back to defaults (fx=fy=320, cx=320, cy=240).  "
        "Pose-based distance estimates will be inaccurate.",
        _CAMERA_CONFIG_SEARCH_PATHS,
    )
    return _DEFAULT_INTRINSICS["fx"], _DEFAULT_INTRINSICS["fy"], \
           _DEFAULT_INTRINSICS["cx"], _DEFAULT_INTRINSICS["cy"]


@dataclass
class TagObservation:
   

    id: int
    center_xy: tuple[int, int]
    side_length_px: int
    est_distance_m: float
    est_yaw_rad: float
    corners: np.ndarray = field(repr=False)

class AprilTagDetector:
  
    FAMILY: str = "tag36h11"

    def __init__(self, tag_size_m: float = 0.065) -> None:
        from pupil_apriltags import Detector  # hard dependency — must be installed

        self._tag_size_m = tag_size_m
        self._fx, self._fy, self._cx, self._cy = _load_intrinsics()
        self._has_intrinsics = not (
            self._fx == _DEFAULT_INTRINSICS["fx"]
            and self._fy == _DEFAULT_INTRINSICS["fy"]
            and self._cx == _DEFAULT_INTRINSICS["cx"]
            and self._cy == _DEFAULT_INTRINSICS["cy"]
        )

        # nthreads=1: deterministic, quad_decimate=2.0: ~2× faster at slight
        # cost to detection range - fine for a 640×480 30 fps loop
        self._detector: "Detector" = Detector(
            families=self.FAMILY,
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        logger.info(
            "apriltags: detector ready  family=%s  tag_size=%.3f m  "
            "intrinsics=(fx=%.1f fy=%.1f cx=%.1f cy=%.1f)  "
            "pose_estimation=%s",
            self.FAMILY,
            self._tag_size_m,
            self._fx, self._fy, self._cx, self._cy,
            "enabled" if self._has_intrinsics else "DISABLED (default intrinsics)",
        )

    def detect(self, bgr_frame: np.ndarray) -> list[TagObservation]:
       
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        raw_tags = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self._fx, self._fy, self._cx, self._cy),
            tag_size=self._tag_size_m,
        )

        observations: list[TagObservation] = []
        for tag in raw_tags:
            if tag.decision_margin < 20:
                # Low-confidence detections tend to have wrong IDs - skip them.
                continue

            corners = tag.corners.astype(int)            # shape (4, 2)
            cx_px = int(tag.center[0])
            cy_px = int(tag.center[1])

            # side_length_px: mean of all four edge lengths
            side_px = self._mean_side_px(corners)

            # Pose-based distance & yaw
            dist_m, yaw_rad = self._pose_to_dist_yaw(tag)

            observations.append(TagObservation(
                id=tag.tag_id,
                center_xy=(cx_px, cy_px),
                side_length_px=side_px,
                est_distance_m=dist_m,
                est_yaw_rad=yaw_rad,
                corners=corners,
            ))

        # Sort: nearest first (inf sorts last, then we flip by side_length_px
        # for those fallback entries so the largest / closest proxy comes first)
        observations.sort(
            key=lambda t: (
                t.est_distance_m if math.isfinite(t.est_distance_m)
                else (1e6 - t.side_length_px)
            )
        )
        return observations

    @staticmethod
    def _mean_side_px(corners: np.ndarray) -> int:
        """Return the mean side length of a quadrilateral (in pixels)."""
        total = 0.0
        n = len(corners)
        for i in range(n):
            dx = corners[(i + 1) % n][0] - corners[i][0]
            dy = corners[(i + 1) % n][1] - corners[i][1]
            total += math.hypot(dx, dy)
        return max(1, int(round(total / n)))

    def _pose_to_dist_yaw(
        self, tag
    ) -> tuple[float, float]:
       
        if tag.pose_t is None:
            return float("inf"), 0.0

        t = tag.pose_t.flatten()   # shape (3,)
        tx, _ty, tz = float(t[0]), float(t[1]), float(t[2])

        dist_m = math.sqrt(tx**2 + _ty**2 + tz**2)

        # Guard against degenerate pose (tag nearly parallel to image plane)
        if tz < 1e-6:
            return dist_m, 0.0

        yaw_rad = math.atan2(tx, tz)
        return dist_m, yaw_rad
