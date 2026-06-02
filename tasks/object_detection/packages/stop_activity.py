from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    # A duckie only matters when it is close: its bottom edge (y2) sits low in
    # the frame. Far-away duckies (small box, high up) are not an immediate
    # threat, so we ignore them.
    stop_line = img_size * 0.6
    for (x1, y1, x2, y2), score, class_id in detections:
        if class_id != 0:          # only duckies are pedestrians to avoid
            continue
        if y2 >= stop_line:
            return True, f"{class_names.get(class_id, 'object')} ahead"
    return False, ""
