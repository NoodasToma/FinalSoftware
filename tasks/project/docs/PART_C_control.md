# Part C — Control & Integration ("Act")

**Owner:** Person C
**Job:** turn Part B's decisions into hardware action, and wire all three layers into one
running robot. This is the **motor/LED layer plus the two main loops** — it *consumes*
Part A (See) and Part B (Decide) and writes the wheels and LEDs.

## Files owned

| File | Role |
|---|---|
| `packages/maneuvers.py` | `ramp_speed`, encoder turns, `_await_motion`, `_straight_exit` |
| `packages/obstacles.py` | `should_stop_for_obstacle` — the brake-for-duckie gate |
| `packages/agent.py` | the **sim** main loop + LED policy (`_set_brake`, `_set_blinker`) |
| `packages/agent_signs.py` | the **bot** main loop (drives Part B's `sign_detection` FSM) |

> Dependency direction: **Act (C) → Decide (B) → See (A)**. Part C imports from both;
> neither imports from C.

---

## 1. `maneuvers.py` — speed ramp + closed-loop turns

### `ramp_speed(current, target, max_step) -> float`
Moves `current` toward `target` by at most `max_step` (smooth accel/decel — no
jump-starts). Snaps to `target` once within one step.

### `_await_motion(...)` — the core of every maneuver
Blocks until a maneuver completes, on **either** platform:
- **Encoder path** (`wheels.encoders` present — real Dagu *and* the sim's modelled
  135-tick/rev encoder): wait until the outer wheel advances `target_ticks` —
  geometry-faithful, independent of CPU/loop speed.
- **Seconds fallback** (`encoders is None`, or ticks never move): rotate for `seconds`
  of wall-clock, with a hard cap so a stuck turn can't freeze the agent.
- **Perception stays alive during the turn:** an optional `tick()` callback runs each
  iteration (same thread → still a single I²C writer). If it reports an obstacle the
  maneuver **HOLDS** (wheels 0) and resumes when clear; held time is subtracted from the
  budget so the arc still completes. `hold_max_s` caps total hold.

### `turn_left / turn_right / straight_through(wheels, stop_event, timings, tick=…, set_wheels=…, hold_max_s=…)`
- `_turn_cfg()` reads **per-direction** params (`turn_left_*` / `turn_right_*`, fallback
  to shared `turn_*`): inner/outer speed (the arc radius) + ticks + seconds.
- Each turn: set wheels → `_await_motion` on the **outer** wheel → `_straight_exit` →
  stop.
- `_straight_exit()` drives straight `turn_exit_ticks` out of the junction box before the
  lane PD takes back over — inside the box there are no clean lane markings, so resuming
  too early makes the PD lock a stray edge and unwind the turn.

Config: `config/maneuver_timings.yaml` (sim) + the `maneuver_timings_bot.yaml` overlay
(bot — PWM-corrected speeds, real-tile tick counts).

---

## 2. `obstacles.py` — `should_stop_for_obstacle(detections, frame_h, frame_w=None, cx_margin_frac=0.0)`

The brake gate, applied to Part A's detections (`[(bbox, score, cls), …]` from YOLO or
`duck_hsv`). Stop if any **duckie** (class 0) is **close** (`y2 > 0.6·frame_h`) **or**
**large** (area `> 0.04·640·frame_h`). Returns `(True, reason)` / `(False, '')`.
Optional "in the way" gate: with `frame_w` + `cx_margin_frac>0`, ignore detections
outside the central band (the bot sets `0.20`; sim leaves `0.0` = off, identical to the
original rule).

---

## 3. `agent.py` — the sim main loop (integration)

`main(camera, wheels, leds, stop_event, *, …)` is the perception → decision → motor loop
(~50 Hz, one control thread). The bare 4-arg call behaves per the original contract; the
keyword-only hooks have no-op defaults:

| Hook | Platform | Purpose |
|---|---|---|
| `observer(snapshot)` | sim | per-loop dict of perceived+decided state → telemetry |
| `frame_observer(bgr)` | both | hand each frame to the server's `/video` (single camera reader) |
| `apriltag_intrinsics` / `apriltag_tag_size` | sim | give the tag detector real metric distance |
| `timings_override` | bot | base timings ⊕ `maneuver_timings_bot.yaml` |
| `lane_config_path` | bot | gentle hardware lane gains |
| `drive_gate()` | sim | skip wheel writes while the dashboard owns the motors |
| `manual_cmd()` | bot | dashboard manual drive — the agent writes it (single I²C writer) |

**Threading (all single-writer-safe on the shared I²C bus):**
- `_IO_LOCK` serialises *every* wheel + LED write (HAT + LED board share one bus;
  concurrent access threw `OSError(121)`).
- A **decoupled duck-detector thread** runs the heavy model off-loop (`maxsize=1` queue);
  the control loop only reads the latest result, so inference never blocks steering.
- An optional **camera watchdog** (off by default — it'd be a second I²C writer).

**Per-frame loop:** read camera → `frame_observer` → AprilTags every Nth frame (Part A) +
enqueue/inline the duck & vehicle detectors (Part A) → `signs = [(o, lookup(o.id)) …]` →
**obstacle debounce** (confirm streak + grace window) → red-line/red-band (Part A) →
accumulate approach signs + **commit latch** → `_derive_event(...)` → `next_state` (Part B)
→ act per state → `observer` snapshot → `sleep(0.02)`.

**`_derive_event`** turns perception into FSM events: obstacle (DRIVE/APPROACH);
`see_stop_or_yield` / `see_light` / `see_intersection` when a relevant sign is "near"
(`est_distance < react_distance` calibrated, or `side_length_px ≥ react_min_px`
uncalibrated); `at_stop_line` in APPROACH (red band/line, or tag pixel size ≥ `sign_stop_px`);
`obstacle_cleared` in SOFT_STOP. The **commit latch** stops the bot if a near sign rose
overhead out of view right at the line.

**Per-state actions:**
- **DRIVE** — `LaneServoingAgent.compute_commands(rgb)` × ramped speed; LEDs off. (In
  `turn_mode: lane_follow`, a chosen junction direction biases the wheels across the box,
  then lane-following resumes.)
- **APPROACH** — `approach_creep: true` (sim) decelerate-and-creep with brake LEDs;
  `false` (bot) keep lane-following at speed and only act at the commit point.
- **STOPPED** — wheels 0, brake LEDs; hold `stop_wait_seconds`; then if
  `_clear_to_enter(...)` **and** `legal_turns` (Part B's `merge_turn_constraints`): pick a
  **random** legal turn and run the maneuver (or resume lane-following). Else → `wait`.
- **WAIT** — hold until `_clear_to_enter` or wait-timeout.
- **TURN_* / STRAIGHT_THROUGH** — blinker on; run the Part-C maneuver (with live
  `_maneuver_tick` so it holds for obstacles); set `sign_cooldown`; back to DRIVE.
- **SOFT_STOP** — hold until the obstacle clears (no timeout — a duckie is a pedestrian).

**`_clear_to_enter(...)`** — the single entry gate: go iff right-of-way
(`we_go_first`, Part B) **and** no obstacle. (The light terms exist but are dormant now
that traffic lights are out of scope — see the note below.)

**LED policy:** `_set_brake` → back LEDs red for APPROACH/STOPPED/WAIT/SOFT_STOP;
`_set_blinker` → 2 Hz yellow on the turning side; DRIVE = off. All guarded by `if leds`.

**Shutdown:** a `try/finally` zeroes the wheels and turns LEDs off on `stop_event`,
Ctrl-C, or any escaped exception.

---

## 4. `agent_signs.py` — the bot main loop (integration)

What the **real bot** actually runs (`real_server.py`, because `maneuver_timings_bot.yaml`
sets `sign_agent: true`). Same `main(...)` signature + dashboard hooks. The loop:
1. read camera; run `detect_duckies_hsv` + `detect_vehicles_hsv` (Part A) every Nth frame.
2. `left, right, sign_state = LaneServoingAgentWithSigns.compute_commands(rgb, veh_dets)`
   — lane command, then Part B's `SignBehaviorFSM` overrides it when handling a sign.
3. layer the **debounced duckie/other-bot soft-stop** on top (`should_stop_for_obstacle`
   + a confirm streak + grace window) — brake whatever the lane wanted.
4. write wheels via the single I²C-locked writer; back LEDs red while braking/handling a
   sign; emit the `observer` snapshot.
`try/finally` leaves the bot safe (wheels 0, LEDs off) on exit.

This is the clean Act-uses-Decide boundary: `agent_signs` owns the loop + motors; the
sign *decisions* live in Part B's `sign_detection/`.

---

## How the three parts connect (one frame)

```
camera.read()
  ├─ AprilTagDetector + lookup ............. Part A (See)
  ├─ duck_hsv / YOLO + vehicle detect ...... Part A (See)
  └─ red_band / red_line ................... Part A (See)
                 │
                 ▼
   _derive_event ► next_state / SignBehaviorFSM ► merge_turn_constraints + we_go_first   ... Part B (Decide)
                 │
                 ▼
   ramp_speed · turn_* · should_stop_for_obstacle · _set_brake/_set_blinker ............... Part C (Act)
                 │
                 ▼
        wheels.set_wheels_speed(...) / leds.set_rgb(...)
```

## Acceptance checks (all pass)
`ramp_speed` math; `should_stop_for_obstacle` close/large/empty; `agent.main` and
`agent_signs.main` import cleanly; `finally` zeroes the motors on stop.

## Out of scope — traffic lights (removed)
`agent.py` still contains the light path (`light.arm/disarm`, `light_was_red`,
`see_light`, the yellow guard, the light terms in `_clear_to_enter`). It still runs, but
since traffic lights were dropped it is **dormant / unowned**. Removing it is a small
refactor across `agent.py` (and the `see_light`/`APPROACH` wiring) if you want the loop
trimmed to signs + obstacles + precedence only.
