"""Minimal `vehicle_detected` ported from the reference's detection.py (only the
piece SignBehaviorFSM.CHECKPATH needs — whether another robot is in view and how far
off-centre it is). We do NOT port the reference's full 416-line detection.py (it pulls
in the ONNX duck model); the project feeds its own object detections in instead."""

from typing import List, Tuple

# Class id of "another vehicle/robot" in the detections fed to the FSM. The project's
# object detector and HSV vehicle detector both emit class 1 for other Duckiebots.
_VEHICLE_CLASS = 1
_VEHICLE_MIN_BBOX_AREA = 800
_DEFAULT_IMG_WIDTH = 640


def vehicle_detected(detections, img_width: int = _DEFAULT_IMG_WIDTH):
    """detections: [((x1,y1,x2,y2), score, cls_id), ...]. Returns (seen, offset)
    where offset is |centre_x - img_centre| / img_width for the first qualifying
    vehicle (class 1, large enough), else (False, 0.0). Mirrors the reference."""
    if not detections:
        return False, 0.0
    for bbox, _score, cls_id in detections:
        if int(cls_id) != _VEHICLE_CLASS:
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area < _VEHICLE_MIN_BBOX_AREA:
            continue
        centre_x = (x1 + x2) / 2.0
        offset = abs(centre_x - img_width / 2.0) / float(img_width)
        return True, offset
    return False, 0.0
