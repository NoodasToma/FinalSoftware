from typing import List, Tuple
import numpy as np


def detect_curve(yellow_xs: List[int], white_xs: List[int],
                 curve_threshold: int = 350) -> Tuple[bool, int]:
    """Is the road CURVING, and which way?

    yellow_xs / white_xs are the per-slice mean x of each lane line, ordered from
    the slice FARTHEST ahead (index 0) to the slice NEAREST the robot (index -1)
    — see detect_lines_in_slices. On a straight the line sits at roughly the same x
    in every slice; on a curve the FAR end shifts sideways relative to the NEAR end.
    So the horizontal shift (far - near) across the slices is the curvature signal.

    Returns (is_curve, direction): direction +1 = road bends RIGHT (far end is to
    the right of the near end), -1 = bends LEFT, 0 = straight. Used by the lane
    agent to slow down and add steering authority through the bend so it doesn't
    drift wide and cross the outer line. Was previously a stub (always straight),
    i.e. the bot had NO curve handling at all.
    """
    best_shift = 0
    for xs in (yellow_xs, white_xs):
        if xs is not None and len(xs) >= 2:
            shift = xs[0] - xs[-1]                 # far slice minus near slice
            if abs(shift) > abs(best_shift):
                best_shift = shift
    if abs(best_shift) > curve_threshold:
        return True, (1 if best_shift > 0 else -1)
    return False, 0
