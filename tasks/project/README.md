# 🦆 Traffic Lights & Traffic Signs — Project Guide

A self‑driving Duckiebot agent that **follows the lane** and **reacts to the
world around it**: it stops at stop signs, picks legal turns at intersections,
obeys traffic lights, brakes for duckies in the road, and yields to another
robot at a junction.

This is the practical "what it does and how to run it" guide. For the **full
behaviour specification** — every road sign and the bot's reaction to it, the
expected behaviour in every situation, and how to deploy + calibrate on the
**real Duckiebot** — see **`BOT_BEHAVIOR.md`**. (The full task breakdown and team
plan live in `docs/PROJECT_PLAN.md`; the per‑module API contracts are in §4 of
that file.)

---

## 1. What the robot actually does

Drop the bot on the track and start the agent. From then on it runs a single
perception → decision → motor loop, ~50 times a second:

| Situation | What the bot does | LEDs |
|---|---|---|
| Open lane | Follows the lane, ramping smoothly up to cruising speed | off |
| Sees a **stop**/**yield** sign (AprilTag) | Ramps down and rolls up to the stop line | back red |
| Stopped at the line | Reads which turns the intersection sign allows, then picks one | back red |
| **Intersection** (4‑way / T / etc.) | Executes a closed‑loop **left / right / straight** maneuver, then resumes | yellow blinker |
| Sees a **`t‑light‑ahead`** sign | Arms the traffic‑light camera detector | — |
| **Red light** (or settled **yellow**) | Holds at the line until it turns **green** | back red |
| **Duckie** in the road | Soft‑stops until the duckie is removed | back red |
| Another **robot** at the junction | Goes first only if its name sorts first; otherwise waits | back red |

When the agent is told to stop (Ctrl‑C / server shutdown) it always cuts the
motors and turns the LEDs off.

---

## 2. How it works (the pipeline)

Everything runs in **one process / one loop**. The server hands the agent four
objects — `camera`, `wheels`, `leds`, `stop_event` — and that's the whole API.

```
            ┌──────────────────────── camera.read() (BGR frame) ────────────────────────┐
            ▼                          ▼                          ▼                       ▼
   AprilTagDetector            TrafficLightDetector       ObjectDetectionAgent      LaneServoingAgent
   (tag36h11 → IDs)            (HSV → red/yellow/green)   (YOLO → duckies)          (PD lane steering)
            │                          │                          │                       │
            ▼                          │                          │                       │
   sign_registry.lookup(id)           │                          │                       │
   → "stop" / "4-way" / …             │                          │                       │
            └──────────────┬──────────┴──────────────┬───────────┘                       │
                           ▼                          ▼                                   │
                   _derive_event(...)  ──────►  STATE MACHINE (states.py)                 │
                           events:  see_stop_or_yield · at_stop_line · obstacle · …       │
                           ▼                                                              │
        DRIVE → APPROACH → STOPPED → {TURN_LEFT|TURN_RIGHT|STRAIGHT_THROUGH} → DRIVE      │
                           │           └→ WAIT (red light / give way) ┘                   │
                           ▼                                                              ▼
                   maneuvers.py (closed-loop turns + speed ramp) ───────────────► wheels.set_wheels_speed()
                                                                                  leds.set_rgb()
```

The **state machine** is the brain. The **perception modules** only answer
questions ("is there a stop sign?", "what colour is the light?"); they never
touch the motors. The **maneuvers** module turns a decision into wheel commands.

### The states

`DRIVE` (lane‑follow) → `APPROACH` (ramp down toward a sign) → `STOPPED` (decide)
→ `TURN_LEFT` / `TURN_RIGHT` / `STRAIGHT_THROUGH` (execute) → back to `DRIVE`.
Two holding states branch off `STOPPED`: `WAIT` (red light or give‑way) and
`SOFT_STOP` (duckie obstacle), each of which returns to driving once clear.

### How signs become meaning

The bot reads **AprilTags** (the `tag36h11` family — the black‑and‑white squares
on Duckietown signs). Each tag ID is looked up in `apriltagsDB.yaml` (700+
entries) to get its meaning: `stop`, `4-way-intersect`, `t-light-ahead`,
`no-left-turn`, a `Vehicle` name, and so on. The intersection logic then
intersects every sign's allowed turns into the legal set the bot may choose
from.

---

## 3. How to run it

> **⚠️ Use Python 3.11.** This repo has two virtualenvs: `.venv` (Python 3.14)
> and `.venv311` (Python 3.11). AprilTag detection depends on `pupil-apriltags`,
> which **only installs on 3.11** here (there is no 3.14 wheel and it won't build
> from source). The default `python` points at `.venv` (3.14) — if you run there,
> the bot lane‑follows but is **blind to every sign and light**. Always run the
> project with `.venv311`.

### In simulation (Godot)

```powershell
# from C:\MyProjects\FinalSoftware
.\.venv311\Scripts\Activate.ps1
python launch.py --sim --task project
```

This launches Godot 4.6 (already cached locally), loads the `project.tscn` track,
and starts the agent against the simulated bot.

* Watch the **Godot window** for the bot driving the course.
* Watch the bot's **camera POV** at `http://localhost:5000/video`.
* Stop with **Ctrl‑C**.

### On the real Duckiebot

```powershell
.\.venv311\Scripts\Activate.ps1
python launch.py --run --bot <bot-name> --task project   # deploy + start
#   live camera: http://<bot-name>.local:5000/video
python launch.py --stop --bot <bot-name>                 # stop cleanly
```

The bot has no hot‑reload — re‑run `--run` after every code change.

---

## 4. What you'll see in the simulation

The `project.tscn` scene contains a tile road loop, the controlled bot with a
forward camera, a **stop sign carrying AprilTag ID 1**, a parking sign, **12
duckies** along the track, and an **NPC robot** that drives a fixed path.

So in sim you can demo:

* ✅ **Lane following** — the bot drives the loop.
* ✅ **Stop sign + intersection turn** — at the tagged stop sign it ramps down,
  halts, and executes a closed‑loop turn.
* ✅ **Duckie obstacle stop** — duckies in the path trigger a soft stop (the YOLO
  model `tasks/object_detection/models/best.onnx` is present and loads).

As of the **Session 4 fidelity rebuild + Session 5 realistic scene** (see
`HANDOFF.md` §S5/§S4) the sim also exercises, via the per‑behaviour suite
`tasks/project/sim_tests/behaviour_suite.py` (12/12):

* ✅ **Red‑line stops** — red stop lines are painted at the 4‑way and the light;
  the bot stops **at the line** (camera red‑line detector, the real Duckietown
  trigger), holds a **full‑stop pause** (`stop_wait_seconds`), then takes a
  **gradual car‑like arc turn** onto the correct outgoing lane (per‑direction
  arc radii, measured landings).
* ✅ **Traffic lights** — a housing+lens light cycling red/green/yellow behind
  its own red line: the bot arms on the nearby `t‑light‑ahead` sign, stops at
  the line, and proceeds. (Plus 18 offline checks in `verify_task2.py`.)
* ✅ **Other robots** — actual Duckiebot models carrying small vehicle‑tag
  plates (the Duckietown way): a parked bot roadside and the moving NPC (front+
  back plates). Detection + `we_go_first` precedence run live. A second
  *negotiating* robot still needs hardware / a second instance.
* ✅ **Dashboard testing** — `http://localhost:5000/` has **“Watch a behaviour”**
  buttons (lane / stop sign / light / obstacle / other robot) that teleport the
  bot to a scenario start while you watch the camera and a **Live bot status**
  panel (state, pose, signs with distances, light colour, red‑line flag,
  obstacle, legal turns).

> **Sim note on turns + distance (Session 4):** the sim now **models the wheel
> encoders** (135 ticks/rev), so `maneuvers.py` uses the **same tick‑based turn
> path as the real bot** (a 90° turn in sim is ~90° on the bot), and it feeds the
> AprilTag detector **calibrated camera intrinsics** so the bot stops using the
> **real `est_distance`** path — not a sim‑only proxy. The time‑based turn path
> remains only as a fallback for a bot whose encoders fail to init.

---

## 5. File map

```
tasks/project/packages/
├── agent.py              ← the main loop: perception → state machine → motors + LEDs
├── states.py             ← State enum + transition table (the brain's wiring)
├── maneuvers.py          ← speed ramp + closed-loop turns (encoder, time-based fallback)
├── obstacles.py          ← "should I stop for this duckie?"
├── precedence.py         ← "do I go first?" (compare robot names)
├── sign_registry.py      ← AprilTag ID → meaning (loads apriltagsDB.yaml)
├── apriltagsDB.yaml      ← the 700+ Duckietown tag database
├── perception/
│   ├── traffic_light.py  ← HSV camera detector: red / yellow / green
│   ├── apriltags.py      ← pupil-apriltags wrapper: frame → tag IDs + distance
│   └── intersection.py   ← "at the stop line?" + merge legal turns
└── config/
    ├── traffic_light_hsv.yaml  ← colour thresholds (tune for your lighting)
    └── maneuver_timings.yaml   ← speeds + turn/straight tick counts (tune on bot)

servers/project/
├── real_server.py        ← runs the agent on the physical Duckiebot
└── virtual_server.py     ← runs the SAME agent against the Godot sim
```

Reused (not modified) from elsewhere in the repo:
`LaneServoingAgent` (lane steering) and `ObjectDetectionAgent` (YOLO duckies).

---

## 6. Tuning knobs

* **Light colours** drift with lighting. Edit thresholds in
  `config/traffic_light_hsv.yaml`. (Red wraps the hue circle, so it uses two
  ranges.)
* **Turn sharpness / distance** is set by `turn_ticks` / `straight_ticks` and the
  inner/outer speeds in `config/maneuver_timings.yaml`. Calibrate on the bot.
* **Duckie stop sensitivity** lives in `obstacles.py` (bottom‑40%‑of‑frame and
  4%‑of‑area thresholds).

---

## 7. Verified status (2026‑06‑02)

Checked on this machine:

* ✅ All project modules import and pass their acceptance smoke tests
  (traffic light, sign registry, AprilTags, intersection, precedence, states,
  maneuvers, obstacles, agent) under **`.venv311`**.
* ✅ Live **AprilTag detection** works in `.venv311` (`pupil-apriltags` installed).
* ✅ **YOLO duckie model** (`best.onnx`) loads via onnxruntime.
* ✅ **Simulation is wired and launchable**: Godot 4.6 cached, TCP camera/wheel
  drivers in place, `virtual_server.py` imports cleanly, `project.tscn` has the
  track + bot + camera + tagged stop sign + ducks + NPC bot.
* ⚠️ The default `.venv` (Python 3.14) **cannot** run sign/light perception
  (`pupil-apriltags` unavailable). Run with **`.venv311`**.
* ⚠️ Traffic lights and tag‑based two‑bot precedence are **not present in the sim
  scene** — verified by unit tests / intended for real hardware.

`real_server.py` only runs on the actual Duckiebot (it needs the Pi‑only
`smbus2` I²C library); this is expected, not a bug.
