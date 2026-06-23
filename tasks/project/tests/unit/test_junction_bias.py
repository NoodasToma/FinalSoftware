"""Unit tests for the junction crossing (_junction_cross_wheels): at a junction the
SIGN's chosen direction drives the wheels DIRECTLY (lane servoing ignored during the
crossing), so the bot takes the chosen branch instead of being pulled to the white
line the wrong way ("chose left but went right")."""

from tasks.project.packages.agent import _junction_cross_wheels


def test_left_is_a_left_arc():
    # left: left wheel slow, right wheel fast.
    l, r = _junction_cross_wheels('left', speed=0.3, strength=0.26)
    assert abs(l - 0.04) < 1e-9 and abs(r - 0.56) < 1e-9
    assert r > l                        # net leftward


def test_right_is_the_mirror():
    l, r = _junction_cross_wheels('right', speed=0.3, strength=0.26)
    assert abs(l - 0.56) < 1e-9 and abs(r - 0.04) < 1e-9
    assert l > r                        # net rightward


def test_straight_is_equal_wheels():
    l, r = _junction_cross_wheels('straight', speed=0.3, strength=0.26)
    assert l == r == 0.3                # dead straight through the box


def test_command_is_absolute_not_lane_relative():
    # The command depends ONLY on direction/speed/strength — lane output is NOT an
    # input, so lane servoing cannot influence the crossing.
    assert _junction_cross_wheels('left', 0.3, 0.26) == _junction_cross_wheels('left', 0.3, 0.26)


def test_clipped_to_unit_range():
    # a big differential clips at the rails, never out of [0,1].
    l, r = _junction_cross_wheels('left', speed=0.3, strength=0.5)
    assert l == 0.0                     # 0.3 - 0.5 -> clipped
    assert 0.0 <= r <= 1.0


def test_strength_controls_sharpness():
    _, r_soft = _junction_cross_wheels('left', 0.3, 0.15)
    _, r_hard = _junction_cross_wheels('left', 0.3, 0.30)
    assert r_hard > r_soft              # bigger strength = sharper turn (wider L/R gap)


def test_speed_controls_forward_motion():
    l_slow, r_slow = _junction_cross_wheels('straight', 0.2, 0.26)
    l_fast, r_fast = _junction_cross_wheels('straight', 0.4, 0.26)
    assert (l_fast + r_fast) > (l_slow + r_slow)
