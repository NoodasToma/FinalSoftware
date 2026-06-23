# 🤖 Bot Behavior Specification — signs, situations, and the real Duckiebot

What the robot is supposed to do in every situation it can meet, which road signs
exist and how it reacts to each, and exactly what changes (and what doesn't) when
the same code runs on the **real Duckiebot** instead of the simulator.

Everything in this file is grounded in the code: file references are given so a
claim can be checked, and anything the bot does NOT handle is listed honestly.

Companions: `README.md` (overview + how to run) · `SIM_TESTS.md` (test matrix) ·
`HANDOFF.md` (engineering history, fidelity model §S4–S5).

---

## 1. How this works on the actual bot

**The same agent runs on both platforms.** `servers/project/real_server.py` and
`servers/project/virtual_server.py` both call the identical
`tasks/project/packages/agent.main(camera, wheels, leds, stop_event)` — only the
driver objects differ. There is no "sim logic" vs "bot logic"; there are only
drivers and config values.

```powershell
# deploy + start on the robot                # stop cleanly
python launch.py --run --bot <name> --task project
python launch.py --stop --bot <name>
# live camera while running: http://<name>.local:5000/video
```

| Subsystem | Simulator | Real Duckiebot | Same code path? |
|---|---|---|---|
| Wheels | `GodotWheelsDriver` (TCP→Godot) | `DaguWheelsDriver` (PWM motors) | ✅ same `set_wheels_speed` API |
| Encoders | **simulated** 135 ticks/rev (integrates commands) | **real** Hall encoders, 135 ticks/rev | ✅ `maneuvers` uses the tick path on both |
| Camera | Godot viewport → JPEG/TCP, 640×480 | IMX219 CSI camera, 640×480 | ✅ same `camera.read()` BGR frames |
| LEDs | `None` (no-op) | `LEDDriver` (real LEDs) | ✅ agent guards `leds is None` |
| AprilTags | sim intrinsics fx≈252, 0.13 m tags (passed by the sim server only) | file-searched intrinsics, **0.065 m** real tags (defaults) | ✅ same detector; different calibration |
| Loop rate | ~50 Hz | ~50 Hz (Pi CPU permitting) | ✅ |

### 1a. The two camera modes (this decides two sub-paths)

`AprilTagDetector` looks for `fx/fy/cx/cy` in
`duckiebot/camera_driver/config/camera_config.yaml` (or `config/camera_intrinsics.yaml`).

* **Calibrated** (intrinsics present): tag distances are real →
  the bot reacts to stop/yield/light signs only within `sign_react_distance_m`
  (1.5 m), arms the light detector only for a close t-light tag, and can stop on
  `est_distance < stop_distance_m` (0.25 m) as a backup to the red line.
* **Uncalibrated** (⚠️ the repo's `camera_config.yaml` currently has NO fx/fy —
  this is the real bot's mode until someone adds calibration): `est_distance=inf`
  → the bot reacts to signs **on sight**, and the stop backup is the pixel-size
  proxy (`side_length_px > 38`, `intersection.py`). **The red stop line is the
  primary stop trigger in both modes**, so stopping behaviour is correct either
  way; calibrated mode is just more precise and less eager at long range.

In practice a real camera decodes a 6.5 cm tag at roughly 0.5–0.8 m, so even
"react on sight" engages at a sane distance on hardware (the 1.5 m gate exists
because the *sim* decodes tags out to ~3 m).

### 1b. Hardware configuration — what's pre-set vs what you calibrate on site

**Per-platform configs (already wired).** The base YAMLs hold the SIM-verified
values; `real_server.py` (the only hardware entry point) overlays the bot
starting values at startup and prints what it applied:

| File | Used by | Contains |
|---|---|---|
| `tasks/project/packages/config/maneuver_timings_bot.yaml` | bot only (merged over the base) | PWM-corrected turn arcs + real-tile tick counts (below); uncalibrated-camera sign gates (`sign_react_min_px`, `sign_stop_px`, `line_straight_px`); duck/red/hold knobs |
| `config/lane_servoing_config_bot.yaml` | bot only (replaces the lane config) | gentle starting gains (`p 0.25, d 0.1, base 0.3`) |
| `config/lane_servoing_hsv_config_bot.yaml` | bot only (applied via `set_hsv_bounds` at startup) | venue lane HSV (wider yellow so the chunky dashes register) + chunky-dash edge fill |
| `config/camera_intrinsics.yaml` | bot (read by the tag detector) | all-commented until measured → bot runs in safe *uncalibrated* mode |

**Uncalibrated-camera sign behaviour (no intrinsics ⇒ tag distance is ∞).** With no
metric distance, the bot gates sign reactions on the AprilTag's apparent **pixel
size** instead (rock-solid, grows monotonically as it nears the sign), because the
painted stop line here is desaturated orange at an angle and the red-line detector
can't see it. `sign_react_min_px` (start APPROACH when the tag is this big ≈ near),
`line_straight_px` (creep DEAD STRAIGHT at the sign — the lane markings curve away
at the junction so lane-following would drift the bot off it), `sign_stop_px` (stop
= AT the line). Backstop: if the tag clips off the top of the frame (sign overhead)
before reaching `sign_stop_px`, the lost-sign commit stops the bot at the line too.
Calibrate from the live `/telemetry` (each tag's `side_px`) — see §1b step 6.
Calibrating camera intrinsics (step 5) restores the metric path and these are
ignored.

All three deploy with `launch.py --run` (the deploy tarball = `tasks/<task>/packages`
+ `config/` + `servers/<task>` — note `duckiebot/…/camera_config.yaml` is NOT
deployed, which is why intrinsics live in `config/camera_intrinsics.yaml`).

**Why the bot's turn values differ from the sim's** (the part that bit us):
the real `DaguWheelsDriver` maps a wheel command v to **PWM = v·195 + 60**
(`pwm_min=60` overcomes stiction; the sim runs `pwm_min=0`). The +60 floor
**compresses speed ratios toward 1**, so the sim's arc pair 0.36/0.45 would run
much wider on hardware. The overlay's `0.30/0.45` (right) and `0.35/0.45` (left)
reproduce the sim-verified PWM ratios; ticks are scaled ×0.975 for real 0.585 m
tiles. Motors aren't truly linear near the floor — treat these as principled
starting points and run the one-trial correction below.

**On-site calibration, in this order:**

1. **Drive straight (trim).** `config/modcon_config.yaml` → `trim` (now 0).
   Command both wheels equal (run the task on a straight); if it pulls
   left/right, adjust trim in small steps (±0.01) until straight.
2. **Lane HSV** (`config/lane_servoing_hsv_config.yaml`) — until the masks are
   clean under venue lighting, nothing downstream matters.
3. **Lane gains** (`config/lane_servoing_config_bot.yaml`) — raise `p_gain`
   until it tracks center without weaving (typical 0.2–0.5), add `d_gain` only
   against overshoot, raise `base_speed` last.
4. **Red HSV** — verify on site with the laptop tool against the live stream:
   ```powershell
   .venv311\Scripts\python.exe tasks\project\sim_tests\calibrate_real_bot.py `
       --url http://<bot>.local:5000/video --hsv
   ```
   Expect `red_line=True` with a line at the bot's feet and the light colour
   correct at ~0.5–1 m. Tune `red_*`/`line_*` + colour bands in
   `tasks/project/packages/config/traffic_light_hsv.yaml`, redeploy, re-check.
5. **Camera intrinsics** (upgrade from uncalibrated → metric distances). One
   printed 6.5 cm tag + a tape measure, ~2 minutes:
   ```powershell
   # tag exactly 0.50 m in front of the lens, facing it squarely:
   ...calibrate_real_bot.py --url http://<bot>.local:5000/video --measure-fx --distance 0.50
   # paste the printed block into config/camera_intrinsics.yaml, redeploy, then:
   ...calibrate_real_bot.py --url ... --verify --distance 0.75    # expect <10% error
   ```
   (The tool fits fx through the detector's own pose estimate — exact and
   immune to tag-border conventions; validated against the sim's known
   calibration to ~3%. Use `--tag-id` if several tags are visible.)
6. **Turn arcs — one-trial correction.** Place the bot at a stop line, let it
   take each turn (or trigger maneuvers directly), measure the landing:
   * Rotated θ instead of 90° → `new_ticks = ticks × 90 / θ`.
   * Arc too **wide** (overshoots the far lane) → **lower** that direction's
     `*_inner_speed` by ~0.03; too **tight** → raise it. Re-measure once.
   * `straight_ticks`: landed short/long of one tile → scale linearly.
   * **Post-turn drift** (turns ~90° fine, then drifts off the outgoing lane as
     DRIVE resumes) → the arc ends inside the markingless junction box, so the
     lane PD locks a stray edge and unwinds the turn. `turn_exit_ticks` (292 on
     the bot) drives straight out of the box onto the outgoing lane's real
     markings first: exits **short** of the lane → raise it; **overruns** the
     lane → lower it. Per-direction overrides `turn_left_exit_ticks` /
     `turn_right_exit_ticks` exist if one arc lands deeper than the other; set 0
     to disable the exit entirely.

**First-run protocol:** bot on a straight with no signs → clean lane-following →
add one stop sign + red line → stop-at-line + pause + turn → then the light →
then a second robot's plate. One new element at a time — the same order the
behaviour suite proves in sim.

---

## 2. Road signs — the complete catalog and the bot's reaction to each

Signs are AprilTags (`tag36h11`). Each detected ID is looked up in
`apriltagsDB.yaml` (552 entries) → a `kind` the agent reacts to
(`sign_registry.py`). Counts/IDs below are from the DB as shipped.

### Signs that change behaviour

| Sign (`traffic_sign_type`) | Tag IDs | Bot's reaction | Where in code |
|---|---|---|---|
| **stop** | 1, 20–38, 162–196 | DRIVE→APPROACH (creep to the red line) → full stop → wait `stop_wait_seconds` → pick a legal turn → arc through | `agent._derive_event`, `states.py` |
| **yield** | 2, 39 | Same approach-and-stop path as stop; at the line it proceeds only when the entry gate is clear (no light/robot/obstacle blocking) | same |
| **t-light-ahead** | 74–94, 200–230 | Approach + stop at the light's red line; **arms the colour detector** (when the tag is within `light_arm_distance_m`); then red=wait, green=go, settled-yellow=wait | `agent.py` arm gate, `_clear_to_enter` |
| **4-way-intersect** | 8, 13–19, 44–56, 197–199, 231–234, 262–264 | At the stop: legal turns = {left, right, straight} | `sign_registry._TURN_MAP`, `merge_turn_constraints` |
| **T-intersection** (stem of a T) | 11, 65–68, 142–151, 236, 239, 243, 246–247 | legal = {left, right} (no straight — there's no road ahead) | same |
| **right-T-intersect** | 9, 57–60, 132–141, 235, 238, 241, 244, 260–261 | legal = {straight, right} | same |
| **left-T-intersect** | 10, 61–64, 152–161, 237, 240, 242, 245, 248–249 | legal = {straight, left} | same |
| **oneway-right / oneway-left** | 6, 42 / 7, 43 | legal = {right} only / {left} only | same |
| **no-right-turn / no-left-turn** | 3, 40 / 4, 41 | removes right / left from whatever else is legal | `merge_turn_constraints` |
| **do-not-enter** | 5, 69 | legal = ∅ — the bot will NOT enter (see §3 edge cases) | same |
| **Vehicle plate** (`tag_type: Vehicle`) | 400–439 (`megabot01…40`) | Remembered (last 10 seen); at an intersection entry the bot **yields if the other robot's name sorts before its own hostname**, else goes first | `precedence.we_go_first`, `_clear_to_enter` |

Multiple signs at one intersection **intersect** their allowed sets (e.g.
4-way + no-left-turn → {right, straight}). Signs seen at any point during the
approach are accumulated, so a sign that scrolls out of view before the line
still counts.

### Signs that are recognized but have NO special behaviour (honest list)

| Sign | Tag IDs | What happens |
|---|---|---|
| **duck-crossing** | 95–124 | Decoded + named, nothing else. Actual duckies on the road are handled by the YOLO obstacle stop regardless of signage. |
| **pedestrian** | 12, 70–73 | Decoded + named only (same reasoning). |
| **parking** | 125–131 | Decoded + named only — no parking maneuver exists. |
| **StreetName** (112), **Localization** (100), blank (47) | various | Looked up, `kind=''` → ignored by the FSM. |

---

## 3. Situation → expected behaviour ("when we run it, the bot should…")

The brain is the FSM in `states.py`:
`DRIVE → APPROACH → STOPPED → {TURN_LEFT | TURN_RIGHT | STRAIGHT_THROUGH} → DRIVE`,
with `WAIT` (red light / give-way) and `SOFT_STOP` (obstacle) branching off.
Below, **bold** = what you should observe.

> **⚠️ Two real-bot behaviour switches (config, default differs sim vs bot) changed
> rows 4–9 below for the robot — the sim still does the classic flow:**
> 1. **No sign-approach slowdown** (`approach_creep: false`, bot overlay). The bot
>    keeps **following the lane at normal speed** when a sign is in view; it does NOT
>    decelerate/creep/brake on sight. It only ACTS on the sign at the **commit point**:
>    the sign is no longer visible after getting close (tag goes overhead ≈ the red
>    line is reached), the tag is **very close** (`sign_stop_px`), or a red line/band
>    is seen. So `APPROACH` on the bot is just "a sign is pending, still driving."
> 2. **Lane-follow turns** (`turn_mode: lane_follow`, bot overlay). At the commit the
>    bot applies the sign's RULES (stop + pause, right-of-way, legal-turn choice) then
>    **resumes visual lane-following** through the junction instead of an open-loop
>    arc — "follow the road with changed rules." The chosen legal turn is still
>    computed + shown in `/telemetry`; only the MOTION is closed-loop on the lane.
>
> The **sim** keeps `approach_creep: true` + `turn_mode: maneuver` (the smooth-stop +
> random-arc rubric demo). Flip either in `maneuver_timings.yaml` / the dashboard
> config editor to see the bot behaviour in the sim.

| # | Situation | Expected behaviour |
|---|---|---|
| 1 | **Open lane** | DRIVE: lane-follows centered (PD on the yellow/white masks), **ramping smoothly** to `base_speed` (no jump-starts: `ramp_max_step` per loop). LEDs off. |
| 2 | **Curve** | Same DRIVE loop — steering follows the masks. No special state. |
| 3 | **Lane lost** (washed out, gap) | Two bounded recovery phases: brief **gentle sweep** toward the side the lane was last seen, then **straight creep** — never a dead freeze, never a pivot-spin. (`visual_lane_servoing/agent.py`) |
| 4 | **Stop/yield sign ahead** (within react distance) | DRIVE→**APPROACH**: back LEDs red, speed ramps down to a creep. In the final `line_straight_distance_m` it **creeps dead straight** (lane paint ends at the box). |
| 5 | **Reaching the red line** | The painted line fills the bottom of the frame → **stops AT the line** (primary trigger). Backups: tag distance < `stop_distance_m`, or sign-lost commit after `stop_commit_grace_s`. |
| 6 | **At the line (stop sign)** | **Full stop for `stop_wait_seconds` (1.5 s)** — always, before even checking right-of-way. |
| 7 | **Choosing the move** | Legal set = intersection of every sign seen on the approach (§2). **Random choice** among legal turns. No intersection sign seen → all three legal. |
| 8 | **Executing the turn** | **Gradual car-like arc** (forward+right tight-ish; left wide; straight crosses the box), timed by **encoder ticks** on both platforms, then a short **straight exit** (`turn_exit_ticks`) that carries the bot out of the markingless box onto the outgoing lane before the PD resumes (else it would lock a stray edge and unwind the turn). Yellow blinker on the turning side (2 Hz); straight = no blinker. Lands on the outgoing lane and resumes DRIVE. |
| 9 | **Just after the intersection** | `sign_cooldown` (4 s): signs are ignored so the bot doesn't re-trigger on the sign it just obeyed. Obstacle stops still act normally (red lines only matter while APPROACHing, so driving over the box's far line does nothing). |
| 10 | **t-light-ahead sign** | Approach + stop at the light's line like a stop sign, and the **colour detector arms**. |
| 11 | **Light is RED** | Holds at the line in **WAIT**, back LEDs red, until green. The red is latched (`light_was_red`) so a glance away can't unlatch it — only seeing **green** releases. |
| 12 | **Light is GREEN** | Goes (after the §6 pause if it just arrived). |
| 13 | **Yellow, just turned** (<0.5 s) | May proceed (can clear the box). |
| 14 | **Yellow, settled** (>0.5 s) | Treated like red: **waits** (`should_brake_for_yellow`). |
| 15 | **Duckie in the lane** (YOLO box low in frame or large) | **SOFT_STOP**: full stop, back LEDs red, holds **until the duckie is removed** — no timeout, ever (a pedestrian). Resumes DRIVE automatically when clear. |
| 16 | **Duckie roadside** (small/high in frame) | Ignored (below the bottom-60%/4%-area thresholds in `obstacles.py`). |
| 17 | **Another robot's plate at the junction** | Plate ID → robot name. **Yields** (WAIT) if that name sorts before this bot's hostname (plain ASCII ordering, uppercase < lowercase); otherwise **goes first**. E.g. host `duckiebot07` yields to `duckiebot03` but goes before `megabot01` ('d' < 'm'). Know your fleet names — rename hosts if the ordering must match a desired priority. |
| 18 | **Obstacle visible while deciding to enter** | The entry gate (`_clear_to_enter`) refuses: **WAITs** until the box is clear. |
| 19 | **do-not-enter (or contradictory signs → empty legal set)** | The bot stops and **stays stopped** — by design it will not enter illegally. It does not search for an alternative route (known limitation: it waits there). |
| 20 | **Ctrl-C / server stop / deploy stop** | Motors **zeroed** and LEDs off in the agent's `finally` — the bot never coasts away on shutdown. |
| 21 | *(sim only)* duck collision | Godot declares game-over and freezes the bot; the wheel server rejects further commands until reset. |

### LED signals (matches `agent.py`)

| LEDs | Meaning |
|---|---|
| All off | DRIVE (cruising) |
| Back red | Braking/standing: APPROACH, STOPPED, WAIT, SOFT_STOP |
| Yellow blink left/right (front+back, 2 Hz) | Turning that direction |
| All off again | Maneuver done, back to DRIVE |

---

## 4. The numbers (current values; every one is a config knob)

| Knob | Value | Meaning | File |
|---|---|---|---|
| `base_speed` / `approach_creep_speed` | 0.25 / 0.25 | cruise / approach creep | `maneuver_timings.yaml` |
| `ramp_max_step` | 0.05 | accel per loop (smooth stops/starts) | " |
| `stop_wait_seconds` | 1.5 | full-stop pause at the line | " |
| `sign_react_distance_m` / `light_arm_distance_m` | 1.5 / 1.5 | react/arm only near the junction (inf-camera ⇒ on sight) | " |
| `stop_distance_m` / `stop_commit_distance_m` / `stop_commit_grace_s` | 0.25 / 0.3 / 1.0 | tag-distance stop backup + lost-sign commit | " |
| `line_straight_distance_m` | 0.6 | creep straight on the final approach | " |
| `turn_right_*` | 0.36/0.45, 530 ticks | right arc R≈0.3 m | " |
| `turn_left_*` | 0.39/0.45, 850 ticks | left arc R≈0.6 m | " |
| `straight_ticks` | 777 | cross the box (~1.1 m) | " |
| `turn_exit_ticks` | 300 (bot 292) | straight exit out of the box after a turn (~0.44 m); per-dir overrides `turn_{left,right}_exit_ticks`; 0 disables | " |
| `sign_cooldown` | 4 s | ignore signs after a maneuver | " |
| obstacle thresholds | bottom 60% of frame, or 4% of area | "duckie in my path" | `obstacles.py` |
| red-line ROI / pixels | bottom 22% of frame, ≥400 px red | "I am AT the line" | `traffic_light_hsv.yaml` (`line_*`) |
| yellow settle | 0.5 s | fresh vs settled yellow | `traffic_light.py` |
| encoder spec | 135 ticks/rev, r=0.0318 m | identical sim + real | `encoder_driver.py` |

---

## 5. Known limitations on the real bot (be honest at the demo)

1. **Turns are open-loop** (tick-calibrated arcs, no mid-turn lane feedback). They
   land on the lane when calibrated, but a big disturbance mid-turn isn't
   corrected until DRIVE resumes.
2. **Precedence is name-ordering only** — no negotiation protocol, no comms; two
   bots with the gate logic could both yield if names/timing align badly. The
   tie-break is deterministic but trivially simple.
3. **One negotiating robot**: other robots are detected via their plates; this
   bot never models the other's intentions.
4. **HSV is lighting-sensitive**: red line + light colours must be re-tuned per
   venue (§1b.3). A washed-out red line silently downgrades stopping to the tag
   backups — verify the line mask on site.
5. **No behaviour** for parking / duck-crossing / pedestrian signs beyond what
   YOLO obstacle stopping already does (§2).
6. **`do-not-enter` dead-ends the bot** (it waits forever rather than rerouting) —
   correct per the rules, but plan the course accordingly.
