# 🧪 Unit & Integration Tests — `project` (Traffic Signs)

Hardware-free, Godot-free tests for the Traffic-Signs task. They run the project's
**real** perception → decision → motor code; only the *edges* are faked:

* **camera** → an in-process `FakeCamera` that serves synthetic BGR frames carrying
  **genuine `tag36h11` AprilTags** (rendered with `cv2.aruco`, decoded by the same
  `pupil_apriltags` backend the bot uses), **painted red stop-lines**, and
  **coloured traffic-light blobs**;
* **wheels / LEDs / encoders** → fakes that record every command.

So the exact code path that runs on the Duckiebot is exercised end-to-end on a
laptop — no simulator, no robot. (The `../sim_tests/` harnesses are different:
those drive the live Godot sim and are for on-screen/on-bot verification. These
`tests/` are the automated unit + integration suite the project goal asks for.)

## Run

> Use the **Python 3.11** venv — `pupil_apriltags` only installs there.

```powershell
# from the repo root
.venv311\Scripts\python.exe -m pip install pytest      # one-time
.venv311\Scripts\python.exe -m pytest                  # whole suite (~8 s)
.venv311\Scripts\python.exe -m pytest tasks/project/tests/unit -q          # units only
.venv311\Scripts\python.exe -m pytest tasks/project/tests/integration -q   # integration only
.venv311\Scripts\python.exe -m pytest -k traffic_light -v                  # one topic
```

`pytest.ini` (repo root) scopes a bare `pytest` to this suite.

## What's covered (mapped to the rubric)

| Rubric item | Tests |
|---|---|
| Recognise/observe all signs; signs have ID numbers | `unit/test_sign_registry.py`, `unit/test_apriltags.py` |
| Use AprilTag software | `unit/test_apriltags.py` (real `tag36h11` decode), every integration test |
| Intersection signs → choose a turn among the legal ones | `unit/test_intersection.py` (merge), `integration/test_agent_loop.py::test_intersection_turn_*` |
| Stop **and** yield signs → approach + stop at the line | `integration/…::test_stop_sign_sim_redline_stops_and_turns[stop\|yield]` |
| Let left/right pass — who goes first? (precedence) | `unit/test_precedence.py`, `unit/test_entry_gate.py` |
| Entry gate: combine light + precedence + obstacle | `unit/test_entry_gate.py` (`_clear_to_enter`) |
| Stopping for obstacles (object detection) | `unit/test_obstacles.py`, `integration/…::test_obstacle_soft_stop_then_clear` |
| Stop on red light / go on green / settled-yellow | `unit/test_traffic_light.py`, `integration/…::test_traffic_light_red_then_green` |
| FSM wiring | `unit/test_states.py` |
| Smooth stops + closed-loop turns (encoder + time paths) | `unit/test_maneuvers.py` |
| **Separate sim vs real configs (camera differs heavily)** | `integration/test_config_separation.py` |

The suite is **129 hardware-free tests** (unit + integration); run them with the
3.11 venv as shown above.

## The two camera paths (why they're tested separately)

The single biggest sim-vs-real difference is the **camera calibration**, and both
paths have an integration test that drives the bot to a stop:

* **`test_stop_sign_sim_redline_stops_and_turns`** — sim path: metric AprilTag
  distance (`config/camera_intrinsics_sim.yaml`) + a painted red stop-line.
* **`test_stop_sign_bot_proximity_stops_and_turns`** — real-bot path: the camera
  ships **uncalibrated** (`config/camera_intrinsics.yaml` is all-commented →
  distance `+inf`), so the bot stops by gating on the tag's **pixel size**
  (`sign_react_min_px` / `sign_stop_px` in `maneuver_timings_bot.yaml`).

`test_config_separation.py` asserts that split exists in the config files and is
wired through the loaders.

## Harness (`conftest.py`)

* **Fakes** — `FakeCamera`, `FakeWheels` (+ `FakeEncoders`), `FakeLEDs`.
* **Frame builders** — `blank_frame`, `add_apriltag`, `add_red_stop_line`, `add_light`.
* **`run_agent(...)`** — runs `agent.main` in a thread, collects the per-loop
  telemetry snapshots, and stops on a predicate (e.g. once a target state appears).
* **`sim_timings` / `bot_timings`** fixtures — the real config files with pauses/
  tick-counts shrunk so the FSM advances quickly (behaviour identical, just faster).

These are **logic/integration** tests by design — the goal was explicitly
"unit and integration tests; we will test on the bot later", so lane-steering
fidelity, real turn geometry, and HSV-under-venue-lighting are validated on the
robot, not here.
