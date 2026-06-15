import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class TagObservation:
    id: int
    center_xy: Tuple[int, int]
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
    def __init__(self, tag_size_m: float = 0.065, intrinsics=None) -> None:
        self._tag_size = tag_size_m
        # `intrinsics`, when given, is an (fx, fy, cx, cy) tuple used directly
        # (skipping the file search). The sim passes its own camera intrinsics
        # this way so est_distance_m is real and the bot takes the SAME
        # est_distance-based stop-line path as hardware. Default None reproduces
        # the original behaviour exactly: search the camera-config files (the
        # real bot finds its calibration; the sim, lacking one, gets inf).
        self._intrinsics = intrinsics if intrinsics is not None else _try_load_intrinsics()
        self._warned = False
        # AprilTag backend. Try each compatible C-extension in turn and use the
        # first that imports. pupil_apriltags (dev/sim) and dt_apriltags (the
        # library pre-installed on real Duckiebots) share the SAME API
        # (Detector(families=...), detect(gray, estimate_tag_pose, camera_params,
        # tag_size) -> detections with .tag_id/.center/.corners/.pose_t/.pose_R),
        # so detect() below works unchanged with either. If NONE is installed we
        # degrade gracefully to "no tags" instead of crashing the whole agent
        # (the bot still lane-follows + brakes for obstacles, just blind to signs).
        self._det = None
        self._backend = None
        _last = None
        for _modname in ("pupil_apriltags", "dt_apriltags"):
            try:
                _mod = __import__(_modname)
                self._det = _mod.Detector(families="tag36h11")
                self._backend = _modname
                print(f"[AprilTagDetector] using {_modname}")
                break
            except Exception as exc:
                _last = exc
        if self._det is None:
            print(f"[AprilTagDetector] no AprilTag library available "
                  f"(tried pupil_apriltags, dt_apriltags; last error: {_last}); "
                  f"sign/light detection DISABLED. Install one on the bot, e.g. "
                  f"`pip3 install dt-apriltags`.")

    def detect(self, bgr_frame: np.ndarray) -> List[TagObservation]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []
        if self._det is None:
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
