from __future__ import annotations

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


class AprilTagDetector:
    def __init__(self, tag_size_m: float = 0.065) -> None:
        import pupil_apriltags
        self._det = pupil_apriltags.Detector(families="tag36h11")
        self._tag_size = tag_size_m

    def detect(self, bgr_frame: np.ndarray) -> list[TagObservation]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        raw = self._det.detect(gray)

        out = []
        for r in raw:
            cx, cy = int(r.center[0]), int(r.center[1])
            corners = r.corners
            side = float(np.mean([
                np.linalg.norm(corners[i] - corners[(i + 1) % 4])
                for i in range(4)
            ]))
            out.append(TagObservation(
                id=int(r.tag_id),
                center_xy=(cx, cy),
                side_length_px=int(side),
                est_distance_m=float("inf"),
                est_yaw_rad=0.0,
            ))

        return out
