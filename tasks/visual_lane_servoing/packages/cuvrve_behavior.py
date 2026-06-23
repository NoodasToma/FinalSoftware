from typing import List, Optional, Sequence, Tuple
import numpy as np


def detect_curve(yellow_slices: Sequence[Optional[float]],
                 white_slices: Sequence[Optional[float]],
                 curve_threshold: int = 60) -> Tuple[bool, int]:
    """Is the road CURVING, and which way?

    ``yellow_slices`` / ``white_slices`` are SLICE-ALIGNED per-slice mean x of each
    lane line (``None`` where that line wasn't seen in a slice), ordered from the
    slice FARTHEST ahead (index 0) to the slice NEAREST the robot (index -1).

    CURVATURE SIGNAL = the far-vs-near shift of the LANE CENTRE (midpoint of the two
    lines), NOT of an individual line. This matters: a single line CONVERGES toward
    the image's vanishing point as it recedes (pure perspective), so on a perfectly
    STRAIGHT road each line's far-near shift is large (~100 px on the real 160° FOV
    camera) — the old per-line test mistook that for a curve and slammed curve_boost,
    swerving the bot off straights. The two lines converge SYMMETRICALLY, so their
    MIDPOINT barely moves on a straight (~15 px measured) and only shifts on a real
    bend. Using the centre cancels perspective and makes the threshold meaningful.

    Returns (is_curve, direction): +1 = bends RIGHT (far centre right of near),
    -1 = bends LEFT, 0 = straight. Needs BOTH lines in >= 2 common slices to form a
    centre; if only one line is visible we can't cancel perspective, so we report
    STRAIGHT (conservative — better to under-boost than to false-swerve)."""
    centres = [(y + w) / 2.0
               for y, w in zip(yellow_slices, white_slices)
               if y is not None and w is not None]
    if len(centres) < 2:
        return False, 0
    shift = centres[0] - centres[-1]               # far centre minus near centre
    if abs(shift) > curve_threshold:
        return True, (1 if shift > 0 else -1)
    return False, 0
