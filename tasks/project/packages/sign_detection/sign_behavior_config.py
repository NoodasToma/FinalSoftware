"""
sign_behavior_config.py
"""

from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39
    # Added for this project: the Godot sim's 4-way intersection sign is tag 8 (our
    # apriltagsDB maps 8 -> 4-way-intersect). The reference's bot venue didn't use it.
    FOUR_WAY = 8


_TAG_ID_MAP: Dict[int, TagID] = {
    1: TagID.STOP,

    # 1: TagID.TURN_RIGHT_FWD,
    8: TagID.FOUR_WAY,          # sim 4-way-intersect sign (added for this project)
    9: TagID.TURN_RIGHT_FWD,
    10: TagID.TURN_LEFT_FWD,
    11: TagID.TURN_LEFT_RIGHT,

    20: TagID.STOP,
    24: TagID.STOP,
    25: TagID.STOP,
    26: TagID.STOP,

    39: TagID.YIELD,
}


def resolve_tag(raw_id: int) -> Optional[TagID]:
    return _TAG_ID_MAP.get(raw_id)


# Turn tokens use the FSM's vocabulary: "forward" (= straight), "left", "right".
# These are aligned to THIS project's apriltagsDB / intersection semantics:
#   8  = 4-way-intersect   -> forward + left + right
#   9  = right-T-intersect -> forward + right
#   10 = left-T-intersect  -> forward + left      ({left, straight})
#   11 = T-intersection    -> left + right
# NOTE vs the verbatim reference: the reference paired these lists with a 10<->11
# detector swap (SWAP_10_11). We decode tags correctly (swap OFF), so the 10 and 11
# lists are set to their true meanings here — i.e. swapped back from the reference,
# which makes them match both our apriltagsDB AND the enum names.
_TAG_TURNS: Dict[TagID, List[str]] = {
    TagID.FOUR_WAY:        ["forward", "left", "right"],
    TagID.TURN_RIGHT_FWD:  ["forward", "right"],          # 9  = right-T
    TagID.TURN_LEFT_FWD:   ["forward", "left"],           # 10 = left-T  ({left, straight})
    TagID.TURN_LEFT_RIGHT: ["left", "right"],             # 11 = T-intersection
}


class State(IntEnum):
    MOVING = auto()
    APPROACHING = auto()
    SLOWING = auto()
    STOPPED = auto()
    CHECKPATH = auto()
    POST_STOP = auto()
    INTERSECT = auto()
    PRE_TURN = auto()
    TURNING = auto()
    EXITING = auto()


class SignBehaviorConfig:
    """
    Config for sign behavior.

    Accepts **kwargs so old code does not crash if it passes old values.
    """

    def __init__(self, **kwargs):
        # Red-line detection
        self.red_strip_frac: float = kwargs.pop("red_strip_frac", 0.32)

        self.red_roi_left: float = kwargs.pop("red_roi_left", 0.12)
        self.red_roi_right: float = kwargs.pop("red_roi_right", 0.9)

        # Bigger = stop later / closer to red line.
        self.red_line_close_y2_ratio: float = kwargs.pop("red_line_close_y2_ratio", 0.3)

        self.red_pixel_frac: float = kwargs.pop("red_pixel_frac", 0.012)
        self.red_min_area: float = kwargs.pop("red_min_area", 60.0)
        self.red_min_width_frac: float = kwargs.pop("red_min_width_frac", 0.12)

        self.red_hsv_low1: Tuple[int, int, int] = kwargs.pop("red_hsv_low1", (0, 100, 70))
        self.red_hsv_high1: Tuple[int, int, int] = kwargs.pop("red_hsv_high1", (12, 255, 255))
        self.red_hsv_low2: Tuple[int, int, int] = kwargs.pop("red_hsv_low2", (168, 100, 70))
        self.red_hsv_high2: Tuple[int, int, int] = kwargs.pop("red_hsv_high2", (179, 255, 255))

        # Important:
        # Do NOT start default intersection behavior when red line is seen without a saved sign.
        self.default_forward_on_red_without_tag: bool = kwargs.pop(
            "default_forward_on_red_without_tag",
            False,
        )

        # After finishing sign behavior, ignore the same red line for a while.
        self.red_ignore_after_frames: int = kwargs.pop("red_ignore_after_frames", 100)

        self.approach_duration = kwargs.pop(
            "approach_duration",
            3,
        )

        self.approach_speed = kwargs.pop(
            "approach_speed",
            0.1,
        )

        # Slow approach after seeing a sign but before red line.
        self.approach_speed_factor: float = kwargs.pop("approach_speed_factor", 0.75)
        self.approach_ramp_factor: float = kwargs.pop("approach_ramp_factor", 0.985)

        # Full stop behavior
        self.stop_hold_frames: int = kwargs.pop("stop_hold_frames", 18)
        self.slow_ramp_factor: float = kwargs.pop("slow_ramp_factor", 0.8)
        self.stopped_speed_threshold: float = kwargs.pop("stopped_speed_threshold", 0.055)

        # CHECKPATH sweep
        self.check_left_frames: int = kwargs.pop("check_left_frames", 8)
        self.check_right_frames: int = kwargs.pop("check_right_frames", 6)
        self.check_turn_speed: float = kwargs.pop("check_turn_speed", 0.22)
        self.check_settle_frames: int = kwargs.pop("check_settle_frames", 6)

        # POST_STOP
        self.post_stop_frames: int = kwargs.pop("post_stop_frames", 12)
        self.post_stop_speed: float = kwargs.pop("post_stop_speed", 0.28)

        # Pre-turn forward creep
        self.preturn_right_frames: int = kwargs.pop("preturn_right_frames", 6)
        self.preturn_left_frames: int = kwargs.pop("preturn_left_frames", 3)
        self.preturn_speed: float = kwargs.pop("preturn_speed", 0.2)

        # Intersection manoeuvre
        self.intersect_forward_frames: int = kwargs.pop("intersect_forward_frames", 40)
        self.intersect_left_frames: int = kwargs.pop("intersect_left_frames", 55)
        self.intersect_right_frames: int = kwargs.pop("intersect_right_frames", 40)

        # for new bot right needs more speed
        self.intersect_forward_speed: Tuple[float, float] = kwargs.pop(
            "intersect_forward_speed",
            (0.4, 0.5),
        )
        self.intersect_left_speed: Tuple[float, float] = kwargs.pop(
            "intersect_left_speed",
            (0.2, 0.5),
        )
        self.intersect_right_speed: Tuple[float, float] = kwargs.pop(
            "intersect_right_speed",
            (0.60, 0.05),
        )

        # Exiting after intersection
        self.exit_speed: float = kwargs.pop("exit_speed", 0.40)
        self.exit_timeout_frames: int = kwargs.pop("exit_timeout_frames", 3)

        for key, value in kwargs.items():
            setattr(self, key, value)