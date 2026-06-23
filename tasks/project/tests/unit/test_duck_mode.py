"""Unit tests for duckie-detection backend selection (_select_duck_mode).

The user's requirement: detect ducks BY THE MODEL so yellow lane dashes are not
detected as ducks (the model is trained on duckies; the colour-based HSV path can
confuse dashes). These pin the selection logic for each mode."""

import pytest

from tasks.project.packages.agent import _select_duck_mode


@pytest.mark.parametrize("model_ok,hsv_allowed,expected", [
    (True,  True,  "model"),   # model preferred when available
    (True,  False, "model"),
    (False, True,  "hsv"),     # fall back to HSV so the bot still brakes for ducks
    (False, False, "off"),
])
def test_auto_prefers_model_then_hsv(model_ok, hsv_allowed, expected):
    assert _select_duck_mode("auto", model_ok, hsv_allowed) == expected


def test_model_mode_uses_model_when_loaded():
    # 'model' guarantees the model — dashes are never detected as ducks.
    assert _select_duck_mode("model", True, True) == "model"
    assert _select_duck_mode("model", True, False) == "model"


def test_model_mode_is_off_if_model_absent_never_hsv():
    # 'model' must NOT silently fall back to HSV (that could see dashes); it turns
    # duck detection OFF instead, so the choice is explicit.
    assert _select_duck_mode("model", False, True) == "off"
    assert _select_duck_mode("model", False, False) == "off"


def test_hsv_mode_forces_hsv():
    assert _select_duck_mode("hsv", True, True) == "hsv"     # even if model is available
    assert _select_duck_mode("hsv", False, True) == "hsv"
    assert _select_duck_mode("hsv", False, False) == "off"   # hsv disabled -> off


def test_unknown_mode_behaves_like_auto():
    assert _select_duck_mode("weird", True, True) == "model"
    assert _select_duck_mode("", False, True) == "hsv"


def test_case_insensitive_and_whitespace():
    assert _select_duck_mode(" MODEL ", True, True) == "model"
    assert _select_duck_mode("Auto", False, True) == "hsv"
