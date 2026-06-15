from typing import Tuple

_DUCKIE_CLASS = 0
_FRAME_W = 640
_BOTTOM_FRACTION = 0.6
_AREA_FRACTION = 0.04


def should_stop_for_obstacle(detections, frame_h: int) -> Tuple[bool, str]:
    """detections: list of ((x1,y1,x2,y2), score, class_id) from
    ObjectDetectionAgent.detect(). Stop if any duckie (class 0) has
    y2 > 0.6*frame_h OR bbox area > 0.04 * (640*frame_h).
    Returns (True, reason) on stop, else (False, '')."""
    if not detections:
        return False, ""

    y_threshold = _BOTTOM_FRACTION * frame_h
    area_threshold = _AREA_FRACTION * (_FRAME_W * frame_h)

    for bbox, _score, class_id in detections:
        if int(class_id) != _DUCKIE_CLASS:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if y2 > y_threshold:
            return True, f"duckie close (y2={y2:.0f} > {y_threshold:.0f})"
        if area > area_threshold:
            return True, f"duckie large (area={area:.0f} > {area_threshold:.0f})"

    return False, ""
