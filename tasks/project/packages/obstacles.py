from typing import Tuple

_DUCKIE_CLASS = 0
_FRAME_W = 640
_BOTTOM_FRACTION = 0.6
_AREA_FRACTION = 0.04


def should_stop_for_obstacle(detections, frame_h: int,
                             frame_w: int = None,
                             cx_margin_frac: float = 0.0) -> Tuple[bool, str]:
    """detections: list of ((x1,y1,x2,y2), score, class_id) from
    ObjectDetectionAgent.detect(). Stop if any duckie (class 0) has
    y2 > 0.6*frame_h OR bbox area > 0.04 * (640*frame_h).
    Returns (True, reason) on stop, else (False, '').

    Optional "in the way" gate: when frame_w is given AND cx_margin_frac > 0, a
    detection whose centre-x falls OUTSIDE the central band (cx_margin_frac ..
    1-cx_margin_frac of the width) is treated as OFF TO THE SIDE — not in our path
    — and ignored, so the bot only soft-stops for obstacles actually ahead of it
    (e.g. a roadside duck no longer brakes the bot). DEFAULT OFF
    (frame_w=None / cx_margin_frac=0.0): the gate is skipped and the result is
    byte-identical to the original close-OR-large rule, so callers that gate the
    far-left _OBSTACLE_BLOCK sentinel (the stop-line clearance checks) are
    unaffected. Only the duck/model detection call sites opt in."""
    if not detections:
        return False, ""

    y_threshold = _BOTTOM_FRACTION * frame_h
    area_threshold = _AREA_FRACTION * (_FRAME_W * frame_h)
    use_cx_gate = bool(frame_w) and cx_margin_frac > 0.0

    for bbox, _score, class_id in detections:
        if int(class_id) != _DUCKIE_CLASS:
            continue
        x1, y1, x2, y2 = bbox
        if use_cx_gate:
            cx = (x1 + x2) / 2.0
            if cx < frame_w * cx_margin_frac or cx > frame_w * (1 - cx_margin_frac):
                continue                      # off to the side, not in our path
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if y2 > y_threshold:
            return True, f"duckie close (y2={y2:.0f} > {y_threshold:.0f})"
        if area > area_threshold:
            return True, f"duckie large (area={area:.0f} > {area_threshold:.0f})"

    return False, ""
