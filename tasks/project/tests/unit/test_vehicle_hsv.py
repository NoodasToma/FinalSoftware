"""Unit tests — the TAG-FREE other-bot detector (perception/duck_hsv.detect_vehicles_hsv).

The local Duckiebots are BLUE and carry no AprilTag plate, so the tag path can't see
them. This detector finds the other bot purely by colour + shape, and — the user's
key requirement — only flags it as an obstacle when it is "in the way": a large,
low, roughly-centred blue blob ahead. A bot off to the side, far away, or up near
the horizon must NOT trigger a stop. These tests pin exactly that.

Blue fill BGR (180, 40, 10) lands inside the detector's default blue HSV band
([95,80,50]..[130,255,255]) — the same dark blue the sim NPC body renders as."""

import cv2
import numpy as np

from tasks.project.packages.perception.duck_hsv import detect_vehicles_hsv

H, W = 480, 640
BLUE = (180, 40, 10)          # BGR; HSV ~ (114, 241, 180), inside the blue band


def _frame():
    f = np.empty((H, W, 3), np.uint8)
    f[:] = (120, 120, 120)    # mid-grey road (S=0, never blue)
    return f


def _blue_box(x1, y1, x2, y2):
    f = _frame()
    cv2.rectangle(f, (x1, y1), (x2, y2), BLUE, -1)
    return f


def test_centred_blue_bot_in_path_is_detected():
    # large, low, centred blue blob = another bot directly ahead.
    dets = detect_vehicles_hsv(_blue_box(270, 300, 370, 440))
    assert dets, "an in-path blue bot was not detected"
    (x1, y1, x2, y2), score, cls = dets[0]
    assert cls == 1, "other-bot detections must be class 1 (vehicle)"
    assert 0.0 < score <= 1.0
    cx = (x1 + x2) / 2.0
    assert W * 0.2 < cx < W * 0.8, "the detected blob should be roughly centred"


def test_blue_bot_off_to_the_left_is_ignored():
    # same size/height, but hugging the LEFT edge = not in our lane -> no stop.
    assert detect_vehicles_hsv(_blue_box(10, 300, 110, 440)) == []


def test_blue_bot_off_to_the_right_is_ignored():
    assert detect_vehicles_hsv(_blue_box(530, 300, 630, 440)) == []


def test_blue_high_near_horizon_is_ignored():
    # a blob whose centre is high in the frame (far ahead / a blue object on a wall),
    # not low on the road in front of us -> rejected by min_cy_frac.
    assert detect_vehicles_hsv(_blue_box(260, 90, 380, 150)) == []


def test_small_far_blue_blob_is_ignored():
    # centred but tiny (below min_bbox_area_frac) = a distant bot, not close enough
    # to brake for.
    assert detect_vehicles_hsv(_blue_box(305, 350, 335, 380)) == []


def test_non_blue_frame_has_no_detection():
    assert detect_vehicles_hsv(_frame()) == []


def test_empty_or_none_frame_is_safe():
    assert detect_vehicles_hsv(None) == []
    assert detect_vehicles_hsv(np.zeros((0, 0, 3), np.uint8)) == []


def test_all_detections_are_vehicle_class():
    dets = detect_vehicles_hsv(_blue_box(270, 300, 370, 440))
    assert dets and all(cls == 1 for _bbox, _score, cls in dets)


# ------------------------------------------------------------------------------
#  SHIPPED real-bot config (bot_hsv_cfg in maneuver_timings_bot.yaml).
#
#  Retuned from real-bot footage where the OLD band MISSED the bot in ~95% of
#  frames AND false-stopped on bare road. Measured on the bot's own camera:
#    * a real Duckiebot body is VIVID blue   -> S ~ 180-220
#    * dark asphalt / shadow reads as a       -> S ~ 85   ("blue" but desaturated)
#      desaturated "blue"
#  so the one clean separator is SATURATION (blue_lower S-floor = 140). These tests
#  pin that the shipped config detects a vivid bot and rejects desaturated dark road
#  at the SAME size/position — i.e. the fix can't silently regress to either failure.
import os
import yaml

_BOT_CFG = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "packages", "config",
    "maneuver_timings_bot.yaml")))["bot_hsv_cfg"]


def _hsv_blob(h, s, v, x1=270, y1=300, x2=380, y2=440):
    """A frame with a low-centred blob of exactly HSV (h,s,v) on grey road."""
    f = _frame()
    patch = np.full((y2 - y1, x2 - x1, 3), (h, s, v), np.uint8)
    f[y1:y2, x1:x2] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
    return f


def test_shipped_cfg_detects_a_vivid_blue_bot():
    # vivid bot body (S=200, H=114, V=90) low + centred + large -> detected.
    dets = detect_vehicles_hsv(_hsv_blob(114, 200, 90), _BOT_CFG)
    assert dets, "shipped bot config failed to detect a vivid in-path Duckiebot"
    assert all(cls == 1 for _b, _s, cls in dets)


def test_shipped_cfg_rejects_desaturated_dark_road():
    # bare asphalt / shadow (S=85) at the SAME size+position must NOT be a bot
    # (this was the empty-road false stop). The S-floor=140 rejects it.
    assert detect_vehicles_hsv(_hsv_blob(114, 85, 80), _BOT_CFG) == []


def test_shipped_cfg_keeps_body_hue_up_to_125():
    # the real body reaches H~123-125; the old upper cap of 120 clipped it and
    # fragmented the mask below the area gate (the bot was never seen). The shipped
    # upper (128) must still detect a blob at H=124.
    assert detect_vehicles_hsv(_hsv_blob(124, 200, 90), _BOT_CFG), \
        "shipped band clips the bot body hue (regression to the missed-bot bug)"
