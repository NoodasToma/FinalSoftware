"""Unit tests — the intersection ENTRY gate (agent._clear_to_enter).

This is where the Task-2/3 right-of-way rules are enforced together: stop on red /
go on green, hold a settled yellow, yield to a higher-precedence robot, and never
enter while an obstacle blocks the box. The agent calls it from STOPPED and WAIT to
decide "is it safe AND legal to go now?". Tested directly with explicit inputs so
each rule — and their conjunction — is verified."""

from tasks.project.packages.agent import _clear_to_enter, _OBSTACLE_BLOCK
from tasks.project.packages.sign_registry import SignSemantic

NOW = 100.0
HOST = "zzz_host"          # large name so any megabot/aaa vehicle sorts before it
FRAME_H = 480


def _vehicle(name):
    return [SignSemantic(kind="", tag_type="Vehicle", vehicle_name=name)]


def _clear(light_was_red=False, light_color=None, yellow_started_at=None,
           vehicles=(), obstacles=()):
    return _clear_to_enter(light_was_red, light_color, yellow_started_at, NOW,
                           HOST, list(vehicles), list(obstacles), FRAME_H)


# ------------------------------------------------------------- nothing blocking
def test_clear_when_nothing_blocks():
    # plain stop sign: no light, no vehicle, no obstacle -> go.
    assert _clear() is True


# ------------------------------------------------------------- traffic light
def test_red_light_blocks_entry():
    assert _clear(light_was_red=True, light_color="red") is False


def test_green_light_after_red_allows_entry():
    assert _clear(light_was_red=True, light_color="green") is True


def test_settled_yellow_blocks_entry():
    # yellow for >0.5 s -> hold (can't be sure of clearing the box).
    assert _clear(light_color="yellow", yellow_started_at=NOW - 1.0) is False


def test_fresh_yellow_does_not_block():
    assert _clear(light_color="yellow", yellow_started_at=NOW - 0.1) is True


# ------------------------------------------------------------- precedence
def test_yield_to_higher_precedence_robot():
    # 'aaa' < 'zzz_host' -> the other robot goes first, we wait.
    assert _clear(vehicles=_vehicle("aaa")) is False


def test_go_when_we_have_precedence():
    # our host sorts first against a larger-named robot -> we go.
    assert _clear_to_enter(False, None, None, NOW, "aaa",
                           list(_vehicle("zzz_robot")), [], FRAME_H) is True


# ------------------------------------------------------------- obstacle
def test_obstacle_in_box_blocks_entry():
    assert _clear(obstacles=_OBSTACLE_BLOCK) is False


# ------------------------------------------------------------- conjunction
def test_all_clear_required_together():
    # green light + we-have-precedence + no obstacle -> only then go.
    ok = _clear_to_enter(True, "green", None, NOW, "aaa",
                         list(_vehicle("zzz")), [], FRAME_H)
    assert ok is True
    # flip ONE condition (add an obstacle) and it must refuse.
    blocked = _clear_to_enter(True, "green", None, NOW, "aaa",
                              list(_vehicle("zzz")), list(_OBSTACLE_BLOCK), FRAME_H)
    assert blocked is False
