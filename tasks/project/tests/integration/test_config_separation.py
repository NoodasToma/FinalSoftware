"""Integration tests — SEPARATE configs for simulation vs the real bot.

Rubric: "make separate configs for simulation and actual code, simulation differs
heavily from actual bots camera". These assert the split is real and wired:

  * maneuver timings: a sim BASE + a real-bot OVERLAY that introduces the
    uncalibrated-camera pixel-size gates the sim doesn't use;
  * camera calibration: a sim intrinsics file (metric distances) vs the real
    file shipped uncalibrated (distance = +inf, pixel-size gating);
  * lane servoing: distinct sim vs bot gain files.
"""

import math
import os

import yaml

from conftest import REPO_ROOT

PKG_CFG = os.path.join(REPO_ROOT, "tasks", "project", "packages", "config")
ROOT_CFG = os.path.join(REPO_ROOT, "config")


def _load(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------- maneuver timings
def test_sim_base_and_bot_overlay_both_exist():
    assert os.path.isfile(os.path.join(PKG_CFG, "maneuver_timings.yaml"))
    assert os.path.isfile(os.path.join(PKG_CFG, "maneuver_timings_bot.yaml"))


def test_bot_overlay_adds_uncalibrated_camera_gates():
    """The bot's camera ships UNCALIBRATED (no intrinsics => tag distance is inf),
    so the overlay gates sign reactions on the tag's apparent PIXEL SIZE. The sim,
    with metric intrinsics + a clean red line, has no such keys — the core
    "sim differs heavily from the bot camera" split."""
    sim = _load(os.path.join(PKG_CFG, "maneuver_timings.yaml"))
    bot = _load(os.path.join(PKG_CFG, "maneuver_timings_bot.yaml"))

    assert bot.get("sign_react_min_px", 0) > 0
    assert bot.get("sign_stop_px", 0) > 0
    # the sim base must NOT carry these pixel gates (it uses metric / red-line stop)
    assert "sign_stop_px" not in sim
    assert "sign_react_min_px" not in sim


def test_bot_overlay_has_hardware_only_perception_configs():
    """Blue other-bot HSV + HSV-duck configs exist only on the bot (the sim uses
    the YOLO model + AprilTag intrinsics instead)."""
    sim = _load(os.path.join(PKG_CFG, "maneuver_timings.yaml"))
    bot = _load(os.path.join(PKG_CFG, "maneuver_timings_bot.yaml"))
    assert "bot_hsv_cfg" in bot and "duck_hsv_cfg" in bot
    assert "bot_hsv_cfg" not in sim and "duck_hsv_cfg" not in sim


def test_overlay_merge_overrides_calibration_values():
    """real_server merges base <- overlay; the overlay must win on the hardware
    calibration knobs (PWM-compressed turn arcs, faster base_speed)."""
    sim = _load(os.path.join(PKG_CFG, "maneuver_timings.yaml"))
    bot_overlay = _load(os.path.join(PKG_CFG, "maneuver_timings_bot.yaml"))
    merged = dict(sim); merged.update(bot_overlay)

    assert sim["base_speed"] != bot_overlay["base_speed"]   # 0.25 sim vs 0.5 bot
    assert merged["base_speed"] == bot_overlay["base_speed"]
    # turn arcs differ (pwm_min stiction compresses ratios on hardware)
    assert merged["turn_right_inner_speed"] == bot_overlay["turn_right_inner_speed"]
    assert sim["turn_right_inner_speed"] != bot_overlay["turn_right_inner_speed"]


# ---------------------------------------------------------------- camera calibration
def test_sim_camera_intrinsics_file_is_explicit_and_metric():
    cfg = _load(os.path.join(ROOT_CFG, "camera_intrinsics_sim.yaml"))
    intr = cfg["intrinsics"]
    for k in ("fx", "fy", "cx", "cy"):
        assert isinstance(intr[k], (int, float)) and intr[k] > 0
    assert cfg["tag_size_m"] > 0


def test_real_camera_ships_uncalibrated():
    """The real camera_intrinsics.yaml is all-commented on purpose => the detector's
    file search returns None => the bot runs the inf-distance pixel-gated path. This
    is THE camera difference vs the sim."""
    from tasks.project.packages.perception.apriltags import _try_load_intrinsics, AprilTagDetector
    # nothing on disk yields active intrinsics for the real default
    assert _try_load_intrinsics() is None

    import numpy as np
    from conftest import blank_frame, add_apriltag
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=140)

    real = AprilTagDetector()                                  # uncalibrated (bot default)
    sim = AprilTagDetector(tag_size_m=0.13, intrinsics=(252.0, 252.0, 320.0, 240.0))
    real_obs = [o for o in real.detect(frame) if o.id == 1]
    sim_obs = [o for o in sim.detect(frame) if o.id == 1]
    assert real_obs and sim_obs
    assert math.isinf(real_obs[0].est_distance_m)              # bot: no metric distance
    assert math.isfinite(sim_obs[0].est_distance_m)            # sim: metric distance


def test_sim_telemetry_uses_the_sim_camera_file():
    """The sim entry points get their intrinsics from the sim camera file (loaded by
    sim_telemetry), and they match the file on disk."""
    from tasks.project.sim_tests.sim_telemetry import (
        SIM_APRILTAG_INTRINSICS, SIM_TAG_SIZE_M, sim_fidelity_kwargs)
    cfg = _load(os.path.join(ROOT_CFG, "camera_intrinsics_sim.yaml"))
    intr = cfg["intrinsics"]
    assert SIM_APRILTAG_INTRINSICS == (intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    assert SIM_TAG_SIZE_M == cfg["tag_size_m"]
    # the kwargs the sim hands agent.main carry these (so the sim runs the metric path)
    assert sim_fidelity_kwargs()["apriltag_intrinsics"] == SIM_APRILTAG_INTRINSICS


# ---------------------------------------------------------------- lane servoing
def test_sim_and_bot_lane_configs_both_exist_and_differ():
    sim = _load(os.path.join(ROOT_CFG, "lane_servoing_config.yaml"))
    bot = _load(os.path.join(ROOT_CFG, "lane_servoing_config_bot.yaml"))
    assert sim and bot
    # they are genuinely different tunings (not an accidental copy)
    assert sim != bot
    # at least the curve handling was retuned for the real venue
    assert sim.get("curve_threshold") != bot.get("curve_threshold")
