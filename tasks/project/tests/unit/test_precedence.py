"""Unit tests — give-way / "which one goes first?" (precedence.py).

Rubric: "At stop and yield signs, let traffic from left and right pass ... Which
one goes first?". The bot yields to any detected other robot whose name sorts
before its own; otherwise it goes first."""

from tasks.project.packages.precedence import we_go_first
from tasks.project.packages.sign_registry import SignSemantic


def _vehicle(name):
    return SignSemantic(kind="", tag_type="Vehicle", vehicle_name=name)


def test_no_other_robot_means_we_go():
    assert we_go_first("duckiebot07", []) is True


def test_yield_to_lexicographically_smaller_name():
    # 'duckiebot03' < 'duckiebot07' -> the other goes first, we wait.
    assert we_go_first("duckiebot07", [_vehicle("duckiebot03")]) is False


def test_go_first_when_our_name_is_smaller():
    # host 'duckiebot07' vs 'megabot01': 'd' < 'm' -> we go first.
    assert we_go_first("duckiebot07", [_vehicle("megabot01")]) is True


def test_any_smaller_name_makes_us_yield():
    signs = [_vehicle("zzz"), _vehicle("aaa"), _vehicle("mmm")]
    assert we_go_first("ddd", signs) is False     # 'aaa' < 'ddd'


def test_all_larger_names_means_we_go():
    signs = [_vehicle("xxx"), _vehicle("yyy")]
    assert we_go_first("ddd", signs) is True


def test_non_vehicle_signs_are_ignored():
    # A road sign that happens to be in the list must not affect precedence.
    road = SignSemantic(kind="stop", tag_type="TrafficSign", vehicle_name=None)
    assert we_go_first("aaa", [road]) is True


def test_vehicle_without_name_is_ignored():
    nameless = SignSemantic(kind="", tag_type="Vehicle", vehicle_name=None)
    assert we_go_first("aaa", [nameless]) is True


def test_uppercase_sorts_before_lowercase():
    # ASCII ordering: 'M' (77) < 'd' (100). Documented in BOT_BEHAVIOR §3 row 17.
    assert we_go_first("duckiebot07", [_vehicle("Megabot")]) is False
