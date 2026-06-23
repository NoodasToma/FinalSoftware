"""Shared pytest fixtures + test doubles for the ``project`` (Traffic Signs) task.

These tests are **hardware-free and Godot-free**: every test runs the project's
REAL perception + decision modules, with the camera/wheels/LEDs/encoders replaced
by in-process fakes and (for integration tests) synthetic camera frames that carry
*real* ``tag36h11`` AprilTags, painted red stop-lines, and coloured traffic-light
blobs. So the same code path that runs on the Duckiebot is exercised end-to-end on
a laptop, with no `pupil_apriltags` mocking and no simulator.

Run (Python 3.11 venv — `pupil_apriltags` only installs there):

    .venv311\\Scripts\\python.exe -m pytest tasks/project/tests -q
"""

import os
import sys
import threading
import time

import numpy as np
import pytest

# --- make the project importable (modules use absolute ``tasks.project...`` imports)
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- make this conftest's helpers importable from tests in unit/ and integration/
#     subfolders via ``from conftest import ...`` (the subfolder, not this dir, is
#     what pytest puts on sys.path by default, so add this dir explicitly).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO_ROOT = _REPO_ROOT


# ===========================================================================
#  Hardware test doubles
# ===========================================================================
class FakeEncoder:
    """One wheel encoder. ``ticks`` advances by ``step`` on every read, so a
    tick-based maneuver (maneuvers.py) completes after a few reads regardless of
    wall-clock — keeping turn tests fast and deterministic. ``step=0`` models a
    DEAD encoder (no ticks ever), which forces maneuvers onto their time fallback."""

    def __init__(self, step: int = 40):
        self._t = 0
        self._step = step

    @property
    def ticks(self) -> int:
        v = self._t
        self._t += self._step
        return v


class FakeEncoders:
    def __init__(self, step: int = 40):
        self.left = FakeEncoder(step)
        self.right = FakeEncoder(step)


class FakeWheels:
    """Records every ``set_wheels_speed`` call. ``encoders`` present => maneuvers
    use the tick path (same as the real DaguWheelsDriver and the sim). Pass
    ``encoders=None`` to model a bot whose GPIO encoders are dead (time fallback)."""

    def __init__(self, with_encoders: bool = True, encoder_step: int = 40):
        self.history = []          # list of (left, right)
        self.left_pwm = 0.0        # read by the agent's telemetry observer
        self.right_pwm = 0.0
        self.encoders = FakeEncoders(encoder_step) if with_encoders else None

    def set_wheels_speed(self, left, right):
        self.left_pwm = float(left)
        self.right_pwm = float(right)
        self.history.append((float(left), float(right)))

    # convenience for assertions
    @property
    def last(self):
        return self.history[-1] if self.history else None

    def moved(self) -> bool:
        """True if any nonzero wheel command was ever issued."""
        return any(abs(l) > 1e-9 or abs(r) > 1e-9 for l, r in self.history)


class FakeLEDs:
    """Records LED state. Mirrors the real LEDDriver surface the agent touches:
    ``set_rgb(index, [r,g,b])`` and ``all_off()``."""

    def __init__(self):
        self.rgb = {}              # index -> [r,g,b] (current state)
        self.history = []          # every (index, [r,g,b]) ever set
        self.all_off_calls = 0

    def set_rgb(self, index, color):
        self.rgb[index] = list(color)
        self.history.append((index, list(color)))

    def all_off(self):
        self.all_off_calls += 1
        self.rgb = {}

    def release(self):
        pass

    @staticmethod
    def _is_red(c):
        return c[0] > 0.5 and c[1] < 0.5 and c[2] < 0.5

    def any_red(self) -> bool:
        """A back/brake LED is currently red."""
        return any(self._is_red(c) for c in self.rgb.values())

    def ever_red(self) -> bool:
        """A red LED was set at any point in the run (brake at APPROACH/STOP/WAIT)."""
        return any(self._is_red(c) for _i, c in self.history)

    def ever_yellow(self) -> bool:
        """A yellow blinker was set at any point (during a turn)."""
        return any(c[0] > 0.5 and c[1] > 0.3 and c[2] < 0.5 for _i, c in self.history)


class FakeCamera:
    """Serves BGR frames to the agent. The frame can be a static array, or a
    callable ``provider(elapsed_seconds) -> frame`` so a scenario can change over
    time (e.g. an obstacle that appears then is removed). ``read()`` returns the
    (ok, frame) tuple the agent expects; ``start``/``stop`` are no-ops."""

    def __init__(self, frame_or_provider):
        self._t0 = time.time()
        if callable(frame_or_provider):
            self._provider = frame_or_provider
        else:
            self._provider = lambda _t: frame_or_provider
        self.frame = None          # also settable directly by a test

    def read(self):
        if self.frame is not None:
            return True, self.frame.copy()
        f = self._provider(time.time() - self._t0)
        return (f is not None), (None if f is None else f.copy())

    def start(self):
        pass

    def stop(self):
        pass


# ===========================================================================
#  Synthetic camera-frame builders (carry REAL detectable cues)
# ===========================================================================
import cv2  # noqa: E402  (after numpy; both are hard deps of the project)

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def blank_frame(h: int = 480, w: int = 640, fill=(120, 120, 120)) -> np.ndarray:
    """A plain BGR frame (mid-grey road by default)."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = fill
    return img


def add_apriltag(frame: np.ndarray, tag_id: int, size_px: int = 120,
                 center=None) -> np.ndarray:
    """Render a REAL ``tag36h11`` marker (white quiet-zone included) into ``frame``.

    The marker is generated with ``cv2.aruco`` and decoded by the project's
    ``pupil_apriltags`` detector unchanged — so this drives the genuine detection
    path, not a mock. ``center`` defaults to the upper-middle of the frame (signs
    sit above the road)."""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 3)
    cx, cy = center
    marker = cv2.aruco.generateImageMarker(_ARUCO_DICT, int(tag_id), int(size_px))
    quiet = int(size_px * 0.25)
    tile = np.full((size_px + 2 * quiet, size_px + 2 * quiet), 255, np.uint8)
    tile[quiet:quiet + size_px, quiet:quiet + size_px] = marker
    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    th, tw = tile_bgr.shape[:2]
    x0 = int(np.clip(cx - tw // 2, 0, w - tw))
    y0 = int(np.clip(cy - th // 2, 0, h - th))
    frame[y0:y0 + th, x0:x0 + tw] = tile_bgr
    return frame


def add_red_stop_line(frame: np.ndarray, thickness_frac: float = 0.18) -> np.ndarray:
    """Paint a saturated red horizontal band across the bottom of the frame — the
    painted Duckietown stop line. Satisfies both the raw red-line pixel count and
    the structural red-band detector (intersection.py)."""
    h, w = frame.shape[:2]
    y0 = int(h * (1.0 - thickness_frac))
    # pure red in BGR
    frame[y0:h, :] = (0, 0, 255)
    return frame


def add_light(frame: np.ndarray, color: str, radius: int = 34, center=None) -> np.ndarray:
    """Draw a filled traffic-light lens blob in the TOP half of the frame (where
    the detector looks). ``color`` in {'red','yellow','green'}; BGR fills chosen to
    land inside the detector's HSV bands."""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 4, h // 4)
    bgr = {"red": (0, 0, 255), "yellow": (0, 230, 230), "green": (0, 230, 0)}[color]
    cv2.circle(frame, center, radius, bgr, -1)
    return frame


# ===========================================================================
#  Integration harness: run the REAL agent loop against fakes
# ===========================================================================
def run_agent(camera, wheels, leds, *, timings, intrinsics=None, tag_size=None,
              on_update=None, max_seconds: float = 8.0, poll: float = 0.02,
              manual_cmd=None):
    """Run ``agent.main`` in a thread against the given fakes and collect the
    per-loop telemetry snapshots.

    ``on_update(snapshots, camera, wheels)`` is polled while running; return True
    from it to stop early (e.g. once a target state is observed). The agent is
    always stopped and joined before returning. Returns the list of snapshots."""
    import tasks.project.packages.agent as agent

    snapshots = []
    lock = threading.Lock()

    def _observe(snap):
        with lock:
            snapshots.append(snap)

    stop_event = threading.Event()
    kwargs = dict(observer=_observe, timings_override=timings)
    if intrinsics is not None:
        kwargs["apriltag_intrinsics"] = intrinsics
    if tag_size is not None:
        kwargs["apriltag_tag_size"] = tag_size
    if manual_cmd is not None:
        kwargs["manual_cmd"] = manual_cmd

    th = threading.Thread(
        target=agent.main, args=(camera, wheels, leds, stop_event),
        kwargs=kwargs, daemon=True, name="AgentUnderTest")
    th.start()
    t0 = time.time()
    try:
        while time.time() - t0 < max_seconds:
            time.sleep(poll)
            if on_update is not None:
                with lock:
                    snaps = list(snapshots)
                if on_update(snaps, camera, wheels):
                    break
    finally:
        stop_event.set()
        th.join(timeout=5)
    with lock:
        return list(snapshots)


def observed_states(snapshots):
    """The set of FSM state names seen across a run."""
    return {s["state"] for s in snapshots}


def wait_for_state(target):
    """An ``on_update`` predicate: stop once any snapshot reports ``target`` state.
    ``target`` may be a single name or an iterable of names (any match)."""
    targets = {target} if isinstance(target, str) else set(target)

    def _pred(snaps, camera, wheels):
        return any(s["state"] in targets for s in snaps)

    return _pred


# ===========================================================================
#  Config / timing helpers
# ===========================================================================
@pytest.fixture
def sim_timings():
    """The sim-base maneuver timings, with durations/distances shrunk so the FSM
    advances through stop->turn quickly in a test (behaviour is identical; only the
    wall-clock pauses and tick counts are smaller). Detectors that need a model or
    extra threads are off so the loop is lean and deterministic."""
    import yaml
    base = os.path.join(REPO_ROOT, "tasks", "project", "packages", "config",
                        "maneuver_timings.yaml")
    with open(base) as fh:
        t = yaml.safe_load(fh)
    t.update({
        "stop_wait_seconds": 0.05,
        "sign_cooldown": 0.2,
        "ramp_max_step": 1.0,        # reach speed immediately
        "base_speed": 0.5,
        "object_detection": False,   # no YOLO model / worker thread
        "bot_hsv": False,            # no blue other-bot HSV
        "duck_hsv": False,
        # tiny maneuvers so a turn completes in a couple of encoder reads
        "turn_left_ticks": 4, "turn_right_ticks": 4, "straight_ticks": 4,
        "turn_exit_ticks": 0,
        "detect_every": 1,           # run detectors every frame for fast reaction
        "light_wait_timeout_s": 100, # don't let the failsafe mask a real WAIT
    })
    return t


@pytest.fixture
def bot_timings(sim_timings):
    """Sim base merged with the REAL-bot overlay (maneuver_timings_bot.yaml), the
    same merge real_server.py does — i.e. the uncalibrated-camera path that gates on
    AprilTag pixel size. Test-speed overrides from ``sim_timings`` are re-applied on
    top so timing stays fast."""
    import yaml
    overlay_path = os.path.join(REPO_ROOT, "tasks", "project", "packages", "config",
                                "maneuver_timings_bot.yaml")
    with open(overlay_path) as fh:
        overlay = yaml.safe_load(fh) or {}
    merged = dict(sim_timings)
    merged.update(overlay)
    # keep the test-speed knobs (overlay may re-introduce slow values)
    merged.update({
        "stop_wait_seconds": 0.05, "sign_cooldown": 0.2, "ramp_max_step": 1.0,
        "object_detection": False, "bot_hsv": False, "duck_hsv": False,
        "turn_left_ticks": 4, "turn_right_ticks": 4, "straight_ticks": 4,
        "turn_exit_ticks": 0, "detect_every": 1, "light_wait_timeout_s": 100,
    })
    return merged
