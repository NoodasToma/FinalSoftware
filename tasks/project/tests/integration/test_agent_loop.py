"""Integration tests — the REAL ``agent.main`` perception->decision->motor loop.

No hardware, no Godot: the camera serves synthetic BGR frames carrying genuine
``tag36h11`` AprilTags, painted red stop-lines, and coloured light blobs; the
wheels/LEDs/encoders are in-process fakes. Each test drives the actual agent loop
(lane servoing + AprilTag detection + FSM + maneuvers + precedence + obstacle stop
wired together) and asserts on the telemetry snapshots it emits.

These map directly onto the Traffic-Signs rubric:
  * recognise signs / stop at the intersection      -> test_stop_sign_*
  * choose a turn among the legal possibilities      -> test_intersection_turn_*
  * stop on red light, go on green                   -> test_traffic_light_red_then_green
  * stop for obstacles in the path                   -> test_obstacle_soft_stop_then_clear
  * sim-vs-real camera paths both stop the bot        -> test_stop_sign_sim_redline / _bot_proximity
  * always leave the bot safe on shutdown            -> test_shutdown_parks_wheels
"""

import time

import cv2
import pytest

from conftest import (
    FakeCamera, FakeWheels, FakeLEDs,
    blank_frame, add_apriltag, add_red_stop_line, add_light,
    run_agent, observed_states, wait_for_state,
)

# Dark-blue fill (BGR) that lands inside detect_vehicles_hsv's blue band — the
# colour the tag-free other Duckiebot renders as. Used by the tagless-bot tests.
_BLUE_BOT_BGR = (180, 40, 10)


def _blue_bot(frame, x1, y1, x2, y2):
    """Paint a blue other-bot blob (NO AprilTag) into a BGR frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BLUE_BOT_BGR, -1)
    return frame


# --------------------------------------------------------------------------- helpers
def _stop_when(*states):
    want = set(states)

    def _pred(snaps, camera, wheels):
        return any(s["state"] in want for s in snaps)
    return _pred


def _first_index(snaps, state):
    for i, s in enumerate(snaps):
        if s["state"] == state:
            return i
    return None


def _turn_states(snaps):
    turns = {"TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"}
    return [s["state"] for s in snaps if s["state"] in turns]


# =========================================================================== open lane
def test_open_lane_stays_in_drive_and_moves(sim_timings):
    """No signs, no line: the bot just lane-follows (DRIVE), ramping its speed up.
    Also checks the shutdown guard parks the wheels + clears the LEDs."""
    cam = FakeCamera(blank_frame())
    wheels = FakeWheels()
    leds = FakeLEDs()
    snaps = run_agent(cam, wheels, leds, timings=sim_timings, max_seconds=1.5)

    assert len(snaps) > 5, "agent loop did not run"
    assert observed_states(snaps) == {"DRIVE"}, "spurious non-DRIVE state with no signs"
    assert any(s["current_speed"] > 0 for s in snaps), "never ramped up to cruise"
    # shutdown safety
    assert wheels.last == (0.0, 0.0)
    assert leds.all_off_calls >= 1


# =========================================================================== stop sign
@pytest.mark.parametrize("tag_id,sign", [(1, "stop"), (2, "yield")])
def test_stop_sign_sim_redline_stops_and_turns(sim_timings, tag_id, sign):
    """SIM camera path: a stop/yield tag + a painted red line. The bot APPROACHes,
    STOPs at the line, then takes a (legal) turn — the full
    DRIVE->APPROACH->STOPPED->TURN cycle, driven by the real red-line detector.
    Runs for BOTH a stop sign and a yield sign ("at stop and yield signs ...")."""
    frame = add_red_stop_line(add_apriltag(blank_frame(), tag_id=tag_id, size_px=120,
                                           center=(320, 150)))
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()
    snaps = run_agent(cam, wheels, leds, timings=sim_timings,
                      on_update=_stop_when("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"),
                      max_seconds=6)

    states = observed_states(snaps)
    assert "APPROACH" in states
    assert "STOPPED" in states
    assert _turn_states(snaps), "never executed a turn after stopping"
    assert leds.ever_red(), "brake LEDs never lit while stopping"


def test_stop_sign_bot_proximity_stops_then_lane_follows(bot_timings):
    """REAL-BOT camera path: an UNCALIBRATED camera (no intrinsics) with a big stop
    tag and NO red line. The bot must still STOP — gating on the tag's pixel size
    (sign_react_min_px / sign_stop_px in the bot overlay) — and then, because the
    bot runs turn_mode='lane_follow', RESUME LANE-FOLLOWING through the junction
    instead of executing a pre-scripted open-loop arc ("follow the road with changed
    rules"). So it reaches STOPPED, then returns to DRIVE, and NEVER enters a TURN
    state."""
    # ~90 px tag side > sign_stop_px (44) so the proximity stop fires; no red line.
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=90, center=(320, 150))
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()

    def _stopped_then_resumed(snaps, camera, wheels):
        si = _first_index(snaps, "STOPPED")
        return si is not None and any(s["state"] == "DRIVE" for s in snaps[si:])

    snaps = run_agent(cam, wheels, leds, timings=bot_timings,
                      on_update=_stopped_then_resumed, max_seconds=6)

    states = observed_states(snaps)
    assert "APPROACH" in states
    assert "STOPPED" in states
    # lane-follow mode: NO blind pre-scripted arc, ever
    assert not _turn_states(snaps), "bot must NOT run an open-loop turn in lane_follow mode"
    # it resumed driving (followed the road) after the stop
    si = _first_index(snaps, "STOPPED")
    assert any(s["state"] == "DRIVE" for s in snaps[si:]), "did not resume lane-following after the stop"
    # the legal turn was still CHOSEN + surfaced (decision visible), rules still applied
    assert any(s.get("turn_mode") == "lane_follow" for s in snaps)


# ====================================================== junction steering bias (lane_follow)
def test_junction_bias_never_steers_illegal_direction_at_left_T(bot_timings, monkeypatch):
    """The fix for "went right at a left-T": in lane_follow mode with junction bias on,
    after the stop the bot crosses biased toward the CHOSEN legal direction — and at a
    left-T (legal {left, straight}) it must NEVER bias RIGHT. Force the choice to 'left'
    so the assertion is deterministic, and confirm the crossing bias is 'left'."""
    import tasks.project.packages.agent as agent_mod
    monkeypatch.setattr(agent_mod.random, "choice", lambda seq: "left")

    t = dict(bot_timings)
    t["junction_lane_bias"] = 0.22
    t["junction_cross_seconds"] = 1.5
    # a left-T tag, big enough to stop on the proximity path; no red line
    frame = add_apriltag(blank_frame(), tag_id=10, size_px=95, center=(320, 150))
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()

    def _crossed(snaps, camera, wheels):
        return any(s.get("junction_cross") is not None for s in snaps)

    snaps = run_agent(cam, wheels, leds, timings=t, on_update=_crossed, max_seconds=6)

    crosses = [s.get("junction_cross") for s in snaps if s.get("junction_cross")]
    assert crosses, "junction bias never engaged after the stop"
    assert all(c == "left" for c in crosses), f"biased a non-left direction at a left-T: {set(crosses)}"
    assert "right" not in crosses           # the bug: must never cross right at a left-T
    # still no open-loop arc, and the chosen turn matches the bias
    assert not _turn_states(snaps)
    chosen = [s.get("chosen_turn") for s in snaps if s.get("chosen_turn")]
    assert chosen and chosen[-1] == "left"


def test_junction_bias_off_by_default_no_cross(bot_timings, monkeypatch):
    """With junction_lane_bias=0 (default), lane_follow does a plain resume — no bias
    window, so junction_cross stays None (sim/back-compat behaviour)."""
    import tasks.project.packages.agent as agent_mod
    monkeypatch.setattr(agent_mod.random, "choice", lambda seq: "left")
    t = dict(bot_timings)
    t["junction_lane_bias"] = 0.0
    frame = add_apriltag(blank_frame(), tag_id=10, size_px=95, center=(320, 150))

    def _stopped_then_drove(snaps, camera, wheels):
        si = _first_index(snaps, "STOPPED")
        return si is not None and any(s["state"] == "DRIVE" for s in snaps[si:])

    snaps = run_agent(FakeCamera(frame), FakeWheels(), FakeLEDs(), timings=t,
                      on_update=_stopped_then_drove, max_seconds=6)
    assert "STOPPED" in observed_states(snaps)           # it did reach the junction
    assert all(s.get("junction_cross") is None for s in snaps)   # but never armed a bias


# =================================================================== intersection turns
def test_intersection_turn_is_within_legal_set(sim_timings):
    """A left-T-intersect sign limits the legal turns to {straight, left}: the bot's
    chosen maneuver must be one of those and NEVER a right turn."""
    frame = add_red_stop_line(add_apriltag(blank_frame(), tag_id=10, size_px=110,
                                           center=(320, 150)))
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()
    snaps = run_agent(cam, wheels, leds, timings=sim_timings,
                      on_update=_stop_when("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"),
                      max_seconds=6)

    turns = set(_turn_states(snaps))
    assert turns, "never chose a turn"
    assert "TURN_RIGHT" not in turns, "chose an ILLEGAL right turn at a left-T"
    assert turns <= {"TURN_LEFT", "STRAIGHT_THROUGH"}
    # the legal-turn set is surfaced in telemetry while STOPPED
    legal = [s["legal_turns"] for s in snaps if s["state"] == "STOPPED" and s["legal_turns"]]
    assert legal and set(legal[0]) == {"left", "straight"}


def test_intersection_constraints_intersect_to_single_turn(sim_timings):
    """Two signs at one junction INTERSECT: left-T-intersect ({straight,left}) +
    no-left-turn (removes left) => only {straight} is legal, so the bot must go
    straight — deterministic proof that conflicting signs are combined correctly."""
    frame = blank_frame()
    add_apriltag(frame, tag_id=10, size_px=90, center=(210, 150))   # left-T-intersect
    add_apriltag(frame, tag_id=4, size_px=90, center=(430, 150))    # no-left-turn
    add_red_stop_line(frame)
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()
    snaps = run_agent(cam, wheels, leds, timings=sim_timings,
                      on_update=_stop_when("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"),
                      max_seconds=6)

    turns = set(_turn_states(snaps))
    assert turns == {"STRAIGHT_THROUGH"}, f"expected straight-only, got {turns}"


# ======================================================================= traffic light
def test_traffic_light_red_then_green(sim_timings):
    """A t-light-ahead tag arms the colour detector; a RED lens holds the bot in
    WAIT at the line, then the lens turns GREEN and the bot proceeds (stop-on-red,
    go-on-green)."""
    def provider(t):
        f = add_red_stop_line(add_apriltag(blank_frame(), tag_id=74, size_px=110,
                                           center=(320, 150)))
        add_light(f, "red" if t < 2.2 else "green")   # generous red window: reach WAIT first
        return f

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()

    def _went_on_green(snaps, camera, wheels):
        # stop once we've both WAITed (red) and later reached a turn (went on green)
        wi = _first_index(snaps, "WAIT")
        if wi is None:
            return False
        return any(s["state"] in {"TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"}
                   for s in snaps[wi:])

    snaps = run_agent(cam, wheels, leds, timings=sim_timings,
                      on_update=_went_on_green, max_seconds=7)

    # the t-light-ahead tag (id 74) must have been recognised — that's what ARMS
    # the colour detector; without it the light-colour telemetry would be vacuously
    # empty and the red/green assertions below would be meaningless.
    assert any(t["id"] == 74 for s in snaps for t in s.get("tags", [])), \
        "t-light-ahead tag was never detected (detector never armed)"
    armed_seen = [s for s in snaps if (s.get("light") or {}).get("armed")]
    assert armed_seen, "light detector never armed"

    assert "WAIT" in observed_states(snaps), "never held at the red light"
    wi = _first_index(snaps, "WAIT")
    later_turn = [s["state"] for s in snaps[wi:]
                  if s["state"] in {"TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"}]
    assert later_turn, "did not proceed after the light turned green"
    # while waiting, the snapshot should show a red light (armed + detected red)
    red_seen = [s for s in snaps[:wi + 1]
                if (s.get("light") or {}).get("color") == "red"]
    assert red_seen, "light was never read as red"
    # and the light must have actually been GREEN by the time it proceeded
    green_after = [s for s in snaps[wi:] if (s.get("light") or {}).get("color") == "green"]
    assert green_after, "proceeded without the light ever turning green"


# ============================================================================ obstacle
def test_obstacle_soft_stop_then_clear(sim_timings):
    """An obstacle (another robot's Vehicle plate, close + centred ahead) triggers a
    SOFT_STOP; once it's removed the bot resumes DRIVE — "stop for objects in the
    path, hold until clear"."""
    timings = dict(sim_timings)
    timings["obstacle_clear_grace_s"] = 0.1     # resume promptly once clear

    def provider(t):
        if t < 2.2:
            # a near, centred Duckiebot plate => _bot_ahead => obstacle
            return add_apriltag(blank_frame(), tag_id=400, size_px=95, center=(320, 220))
        return blank_frame()                    # robot has moved on

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()

    def _stopped_then_cleared(snaps, camera, wheels):
        si = _first_index(snaps, "SOFT_STOP")
        if si is None:
            return False
        return any(s["state"] == "DRIVE" for s in snaps[si:])

    snaps = run_agent(cam, wheels, leds, timings=timings,
                      on_update=_stopped_then_cleared, max_seconds=7)

    states = observed_states(snaps)
    assert "SOFT_STOP" in states, "never stopped for the obstacle"
    si = _first_index(snaps, "SOFT_STOP")
    assert any(s["state"] == "DRIVE" for s in snaps[si:]), "never resumed after it cleared"
    assert leds.ever_red(), "brake LEDs never lit during the soft stop"


# ================================================== tag-free other-bot soft-stop
def test_tagless_other_bot_in_path_soft_stops_then_clears(sim_timings):
    """The user's goal: detect another Duckiebot with NO AprilTag and soft-stop
    when it's IN THE WAY. A centred blue bot blob (no tag at all) must drive the
    bot to SOFT_STOP purely via the blue-HSV detector, then resume DRIVE once it
    moves on — the same obstacle path as a duckie, but tag-free."""
    timings = dict(sim_timings)
    timings.update({"bot_hsv": True, "obstacle_confirm_frames": 1,
                    "obstacle_clear_grace_s": 0.1})

    def provider(t):
        f = blank_frame()
        if t < 2.2:
            _blue_bot(f, 270, 300, 370, 440)   # centred, low, close -> in our path
        return f                                # then gone

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()

    def _stopped_then_cleared(snaps, camera, wheels):
        si = _first_index(snaps, "SOFT_STOP")
        return si is not None and any(s["state"] == "DRIVE" for s in snaps[si:])

    snaps = run_agent(cam, wheels, leds, timings=timings,
                      on_update=_stopped_then_cleared, max_seconds=7)

    states = observed_states(snaps)
    assert "SOFT_STOP" in states, "never soft-stopped for the tag-free blue bot in its path"
    # the stop was tag-free: no AprilTag was ever rendered, so no Vehicle tag drove it
    assert all(not s.get("tags") for s in snaps), "no tag should ever be detected here"
    si = _first_index(snaps, "SOFT_STOP")
    assert any(s["state"] == "DRIVE" for s in snaps[si:]), "never resumed after the bot moved on"


def test_tagless_other_bot_off_to_the_side_does_not_stop(sim_timings):
    """The "in the way" half: a blue bot off to the SIDE (not in our lane) must NOT
    soft-stop us — detect_vehicles_hsv's lateral gate rejects it, so the bot keeps
    driving. This is the guard against braking for parked/passing bots beside the road."""
    timings = dict(sim_timings)
    timings.update({"bot_hsv": True, "obstacle_confirm_frames": 1})

    frame = _blue_bot(blank_frame(), 10, 300, 110, 440)   # hugging the LEFT edge
    cam = FakeCamera(frame)
    wheels, leds = FakeWheels(), FakeLEDs()
    snaps = run_agent(cam, wheels, leds, timings=timings, max_seconds=2.5)

    assert "SOFT_STOP" not in observed_states(snaps), \
        "soft-stopped for a bot off to the side (in-the-way gate failed)"
    assert wheels.moved(), "bot should keep driving past an off-side bot"


# ================================================================= no approach mechanic
def _pending_sign_timings(base):
    """Make a sign 'pending' but never committing: react gate tiny (enter the
    sign-pending state on sight), stop gate unreachable, no metric/lost commit — so
    with a static sign and no red line the bot keeps DRIVING and we can observe the
    speed it holds while a sign is in view."""
    t = dict(base)
    t.update({"sign_react_min_px": 5, "sign_stop_px": 100000,
              "stop_commit_distance_m": 0.0, "line_straight_px": 0,
              "stop_distance_m": 0.0, "sign_react_distance_m": 0.0})
    return t


def test_approach_mechanic_removed_keeps_full_speed(bot_timings):
    """The sign-approach SLOWDOWN is gone (default approach_creep=False): while a
    sign is in view but not yet at the commit point, the bot keeps following the lane
    at FULL speed instead of decelerating to a creep. Proven by comparing the speed
    it holds (telemetry current_speed) with a sign pending, new behaviour vs the
    legacy creep — the new one must be much faster — and it must NOT have stopped."""
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=80, center=(320, 150))

    def hold_speed(approach_creep):
        t = _pending_sign_timings(bot_timings)
        t["approach_creep"] = approach_creep
        cam = FakeCamera(frame)
        snaps = run_agent(cam, FakeWheels(), FakeLEDs(), timings=t,
                          on_update=wait_for_state("APPROACH"), max_seconds=3)
        pending = [s for s in snaps if s["state"] == "APPROACH"]
        assert pending, "sign never registered as pending"
        # static mid-gate sign + no red line => never commits => never stops
        assert "STOPPED" not in observed_states(snaps), "should not stop on a pending sign"
        return max(s["current_speed"] for s in pending)

    new_speed = hold_speed(False)        # no approach mechanic (default)
    creep_speed = hold_speed(True)       # legacy decelerate-and-creep
    assert new_speed >= 0.9 * bot_timings["base_speed"], \
        f"no-approach mode should hold ~full speed, got {new_speed}"
    assert new_speed > creep_speed + 0.1, \
        f"no-approach speed {new_speed} should far exceed legacy creep {creep_speed}"


def test_pending_sign_does_not_brake_until_commit(bot_timings):
    """A detected-but-not-yet-close sign must not light the brake LEDs — the bot is
    just driving. (Brake LEDs are for STOPPED/WAIT/SOFT_STOP, which a pending sign
    never reaches here.)"""
    t = _pending_sign_timings(bot_timings)
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=80, center=(320, 150))
    leds = FakeLEDs()
    run_agent(FakeCamera(frame), FakeWheels(), leds, timings=t,
              on_update=wait_for_state("APPROACH"), max_seconds=2.5)
    assert not leds.ever_red(), "brake LEDs lit for a merely-pending sign (approach mechanic not removed)"


@pytest.mark.parametrize("flag", ["false", "False", False])
def test_approach_creep_string_false_is_respected(bot_timings, flag):
    """A dashboard/config edit can deliver the STRING 'false' (bool('false') is True
    in Python — a footgun). The agent must treat 'false'/'False'/False all as
    no-approach: a merely-pending sign must NOT brake/creep."""
    t = _pending_sign_timings(bot_timings)
    t["approach_creep"] = flag
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=80, center=(320, 150))
    leds = FakeLEDs()
    run_agent(FakeCamera(frame), FakeWheels(), leds, timings=t,
              on_update=wait_for_state("APPROACH"), max_seconds=2.5)
    assert not leds.ever_red(), f"approach_creep={flag!r} was not honoured as False"


# =================================================== manual drive (bot-safe override)
def test_manual_cmd_override_drives_through_the_agent(sim_timings):
    """The bot-safe manual drive: when manual_cmd() returns a command, the AGENT
    writes THAT to the wheels (single I2C writer) instead of its own — so there's
    never a second wheel-writer thread (the OSError(121) bus-crash cause)."""
    def manual_cmd():
        return (0.33, -0.33)

    cam = FakeCamera(blank_frame())
    wheels, leds = FakeWheels(), FakeLEDs()

    def _saw_manual(snaps, camera, w):
        return (0.33, -0.33) in w.history

    run_agent(cam, wheels, leds, timings=sim_timings,
              manual_cmd=manual_cmd, on_update=_saw_manual, max_seconds=3)
    assert (0.33, -0.33) in wheels.history, "agent did not apply the manual command"


def test_manual_cmd_none_lets_agent_drive(sim_timings):
    """manual_cmd() returning None == agent drives normally, never a manual value."""
    cam = FakeCamera(blank_frame())
    wheels, leds = FakeWheels(), FakeLEDs()
    run_agent(cam, wheels, leds, timings=sim_timings,
              manual_cmd=lambda: None, max_seconds=1.5)
    assert wheels.moved(), "agent should drive when manual_cmd is None"
    assert (0.33, -0.33) not in wheels.history


# ============================================================================ shutdown
def test_shutdown_parks_wheels_and_clears_leds(sim_timings):
    """However it ends, the agent's finally-guard must zero the wheels and turn the
    LEDs off — the bot never coasts away or leaves a light on."""
    cam = FakeCamera(add_apriltag(blank_frame(), tag_id=8, size_px=110))
    wheels, leds = FakeWheels(), FakeLEDs()
    run_agent(cam, wheels, leds, timings=sim_timings, max_seconds=1.2)
    assert wheels.last == (0.0, 0.0)
    assert leds.all_off_calls >= 1
    assert not leds.any_red()        # nothing left lit
