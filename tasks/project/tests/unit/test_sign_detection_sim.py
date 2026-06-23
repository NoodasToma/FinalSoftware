"""Unit tests — the project adaptations to the ported reference sign detection that
make it correct for our SIM (and our bot, which reads tags correctly):
  * the 10<->11 swap is OFF (the swap was a KvatiTown-bot quirk; it would invert
    left-T <-> T-intersection on a correctly-reading camera / rendered sim tags);
  * the sim's 4-way sign (tag 8) is recognised as a 4-way (all three turns).

These render real tag36h11 markers (as the Godot scene textures do) and run them
through the ported detector + tag resolver."""

import cv2
import numpy as np

import tasks.project.packages.sign_detection.april_tag as april_tag
from tasks.project.packages.sign_detection.sign_behavior_config import (
    resolve_tag, _TAG_TURNS, TagID,
)

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _detect(tag_id):
    marker = cv2.aruco.generateImageMarker(_DICT, tag_id, 160)
    canvas = np.full((480, 640), 255, np.uint8)
    canvas[150:310, 240:400] = marker
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    tags = april_tag.detect_tags(None, rgb)
    return [t["tag_id"] for t in tags]


def test_swap_is_off_by_default():
    assert april_tag.SWAP_10_11 is False


def test_no_swap_tags_10_and_11_read_as_themselves():
    # With the swap off, a rendered tag 10 reads as 10 (not 11) and vice-versa.
    assert _detect(10) == [10]
    assert _detect(11) == [11]


def test_tag_10_resolves_to_left_turn_not_inverted():
    # 10 -> TURN_LEFT_FWD; if the swap were on it'd wrongly be TURN_LEFT_RIGHT.
    ids = _detect(10)
    assert resolve_tag(ids[0]) == TagID.TURN_LEFT_FWD


def test_four_way_tag_8_recognised_with_all_turns():
    ids = _detect(8)
    assert ids == [8]
    tid = resolve_tag(8)
    assert tid == TagID.FOUR_WAY
    assert set(_TAG_TURNS[tid]) == {"forward", "left", "right"}


def test_stop_and_right_t_still_resolve():
    assert resolve_tag(_detect(1)[0]) == TagID.STOP            # 1 -> STOP
    assert resolve_tag(_detect(9)[0]) == TagID.TURN_RIGHT_FWD  # 9 -> right-T
    assert set(_TAG_TURNS[TagID.TURN_RIGHT_FWD]) == {"forward", "right"}


def test_ref_turn_semantics_match_our_apriltagsdb():
    """REGRESSION: the ported _TAG_TURNS (with the 10<->11 swap OFF) must match THIS
    project's apriltagsDB / intersection semantics for every intersection tag — not
    the reference's swap-compensated lists. Tags 10/11 were the bug: the reference
    paired its lists with the detector swap; with the swap off they must be the true
    meanings. ('forward' in the FSM == 'straight' in our DB.)"""
    from tasks.project.packages.sign_registry import lookup
    from tasks.project.packages.perception.intersection import merge_turn_constraints

    def norm(lst):
        return {"straight" if t == "forward" else t for t in lst}

    for tag in (8, 9, 10, 11):
        ours = set(merge_turn_constraints([lookup(tag)]))
        ref = norm(_TAG_TURNS[resolve_tag(tag)])
        assert ref == ours, f"tag {tag}: ported {ref} != apriltagsDB {ours}"


def test_left_T_and_T_intersection_not_inverted():
    # the exact bug the review caught: tag 10 = left-T = {left, straight};
    # tag 11 = T-intersection = {left, right}. Must NOT be each other.
    assert set(_TAG_TURNS[resolve_tag(10)]) == {"forward", "left"}     # left-T
    assert set(_TAG_TURNS[resolve_tag(11)]) == {"left", "right"}       # T-int


def test_swap_flag_when_enabled_does_swap():
    # The flag still works if a real backwards-reading camera ever needs it.
    april_tag.SWAP_10_11 = True
    try:
        assert _detect(10) == [11]
        assert _detect(11) == [10]
    finally:
        april_tag.SWAP_10_11 = False        # restore the default for other tests
