from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CAMERA_CONFIG_SEARCH_PATHS = [
    "duckiebot/camera_driver/config/camera_config.yaml",
    "/data/config/calibrations/camera_intrinsic/default.yaml",
]

_DEFAULT_FX = 320.0
_DEFAULT_FY = 320.0
_DEFAULT_CX = 320.0
_DEFAULT_CY = 240.0

_intrinsics_warning_issued = False


def _load_intrinsics() -> tuple[float, float, float, float]:
    global _intrinsics_warning_issued
    try:
        import yaml
    except ImportError:
        if not _intrinsics_warning_issued:
            logger.warning("apriltags: PyYAML not found, using default intrinsics")
            _intrinsics_warning_issued = True
        return _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY

    for path in _CAMERA_CONFIG_SEARCH_PATHS:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
            if all(k in cfg for k in ("fx", "fy", "cx", "cy")):
                return float(cfg["fx"]), float(cfg["fy"]), float(cfg["cx"]), float(cfg["cy"])
            if "camera_matrix" in cfg and "data" in cfg["camera_matrix"]:
                d = cfg["camera_matrix"]["data"]
                return float(d[0]), float(d[4]), float(d[2]), float(d[5])
        except Exception:
            continue

    if not _intrinsics_warning_issued:
        logger.warning(
            "apriltags: intrinsics not found in %s, using defaults (fx=fy=320, cx=320, cy=240)",
            _CAMERA_CONFIG_SEARCH_PATHS,
        )
        _intrinsics_warning_issued = True
    return _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY


@dataclass
class TagObservation:
    id: int
    center_xy: tuple[int, int]
    side_length_px: int
    est_distance_m: float
    est_yaw_rad: float
    corners: np.ndarray = field(repr=False)


class AprilTagDetector:
    FAMILY = "tag36h11"

    def __init__(self, tag_size_m: float = 0.065) -> None:
        from pupil_apriltags import Detector

        self._tag_size_m = tag_size_m
        self._fx, self._fy, self._cx, self._cy = _load_intrinsics()
        self._detector = Detector(
            families=self.FAMILY,
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
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
                continue
            corners = tag.corners.astype(int)
            cx_px = int(tag.center[0])
            cy_px = int(tag.center[1])
            side_px = self._mean_side_px(corners)
            dist_m, yaw_rad = self._pose_to_dist_yaw(tag)
            observations.append(TagObservation(
                id=tag.tag_id,
                center_xy=(cx_px, cy_px),
                side_length_px=side_px,
                est_distance_m=dist_m,
                est_yaw_rad=yaw_rad,
                corners=corners,
            ))

        observations.sort(
            key=lambda t: (
                t.est_distance_m if math.isfinite(t.est_distance_m)
                else (1e6 - t.side_length_px)
            )
        )
        return observations

    @staticmethod
    def _mean_side_px(corners: np.ndarray) -> int:
        total = 0.0
        n = len(corners)
        for i in range(n):
            dx = corners[(i + 1) % n][0] - corners[i][0]
            dy = corners[(i + 1) % n][1] - corners[i][1]
            total += math.hypot(dx, dy)
        return max(1, int(round(total / n)))

    def _pose_to_dist_yaw(self, tag) -> tuple[float, float]:
        if tag.pose_t is None:
            return float("inf"), 0.0
        t = tag.pose_t.flatten()
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
        dist_m = math.sqrt(tx**2 + ty**2 + tz**2)
        if tz < 1e-6:
            return dist_m, 0.0
        return dist_m, math.atan2(tx, tz)
