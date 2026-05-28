from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TagObservation:
    id: int
    center_xy: tuple[int, int]
    side_length_px: int
    est_distance_m: float
    est_yaw_rad: float


def _try_load_intrinsics():
    candidates = [
        "duckiebot/camera_driver/config/camera_config.yaml",
        "config/camera_intrinsics.yaml",
    ]
    try:
        import yaml
    except ImportError:
        return None

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
            src = cfg.get("intrinsics", cfg)
            if all(k in src for k in ("fx", "fy", "cx", "cy")):
                return (float(src["fx"]), float(src["fy"]),
                        float(src["cx"]), float(src["cy"]))
        except Exception:
            continue
    return None


class AprilTagDetector:
    def __init__(self, tag_size_m: float = 0.065) -> None:
        import pupil_apriltags
        self._det = pupil_apriltags.Detector(families="tag36h11")
        self._tag_size = tag_size_m
        self._intrinsics = _try_load_intrinsics()
        self._warned = False

    def detect(self, bgr_frame: np.ndarray) -> list[TagObservation]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        if self._intrinsics is None and not self._warned:
            print("AprilTagDetector: no camera intrinsics found, "
                  "est_distance_m will be inf")
            self._warned = True

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        if self._intrinsics is not None:
            fx, fy, cx, cy = self._intrinsics
            raw = self._det.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=self._tag_size,
            )
        else:
            raw = self._det.detect(gray)

        out = []
        for r in raw:
            cx_px, cy_px = int(r.center[0]), int(r.center[1])
            corners = r.corners
            side = float(np.mean([
                np.linalg.norm(corners[i] - corners[(i + 1) % 4])
                for i in range(4)
            ]))

            if self._intrinsics is not None and r.pose_t is not None:
                dist = float(np.linalg.norm(r.pose_t))
                yaw = math.atan2(r.pose_R[0, 2], r.pose_R[2, 2])
            else:
                dist = float("inf")
                yaw = 0.0

            out.append(TagObservation(
                id=int(r.tag_id),
                center_xy=(cx_px, cy_px),
                side_length_px=int(side),
                est_distance_m=dist,
                est_yaw_rad=yaw,
            ))

        return out
