# 🤝 Handoff — `project` sim (SESSION 5 — realistic scene: red-line stops, car-like turns, real bots)

**Updated:** 2026-06-04 · **Branch:** `main`
**Companion docs:** `tasks/project/README.md` (what it does), `tasks/project/BOT_BEHAVIOR.md`
(⭐ behaviour spec: every sign + every situation + real-bot deployment/calibration),
`tasks/project/SIM_TESTS.md` (test matrix), `docs/MAP_MAKER.md` (editing the scene).
(`docs/PROJECT_PLAN.md` is referenced by README but does not exist.)

> ⚠️ **SESSION 5 supersedes scene/config/behavior specifics in ALL sections below** (S4 still describes
> the fidelity model — encoders/intrinsics/telemetry — which is unchanged and still the foundation).

---

## S5.0 — What changed (user asks: normal signs · red-line stops · wait · car turns · real bots · dashboard)

1. **Normal-sized signs, no giant QR billboards.** All sim AprilTags are now **0.13 m** plates
   (`SIM_TAG_SIZE_M=0.13`): the stop octagon carries a small 0.13 tag plate; the billboard scene
   (`obj_tag_billboard.tscn`) is restyled into a small roadside sign (0.16 plate, short post). NOTE
   (measured): Godot's crisp nearest-filter rendering decodes 0.13 tags out to **~3 m** (16 px!), far
   beyond a real camera — which is WHY the distance gates below exist.
2. **The bot stops AT a painted red line** (`detect_red_line` in `perception/intersection.py`): red-HSV
   over the **bottom ~22%** of the frame (camera tilts 14° down → that ROI only sees road within ~0.3 m,
   so it fires exactly at the line; red octagons/lenses sit higher and can't false-trigger). This is the
   PRIMARY stop trigger (the hardware-correct one — real Duckietown stops at red lines); tag distance
   (`stop_distance_m 0.25`) + a grace-gated commit latch (`stop_commit_grace_s 1.0`) are backups only.
   Red line meshes painted at all 4 arms of the 4-way + at the traffic light (`StopLines/` in the map).
3. **Full stop + wait**: `stop_wait_seconds 1.5` — STOPPED holds the pause FIRST, then checks
   right-of-way (pause-first also debounces 1-frame "not clear" blips that used to detour via WAIT).
   After WAIT (e.g. red light) clears, no second pause.
4. **Car-like gradual turns** (per-direction arc params in `maneuvers._turn_cfg`, fallback to legacy
   keys): right = forward+right arc R≈0.3 (`turn_right_inner 0.36/outer 0.45/530 ticks`), left = wide
   arc R≈0.6 (`0.39/0.45/850`), straight `777 ticks`. **Measured landings (3 runs, identical):** right →
   (6.31, 5.86) Δh −91° = east-arm lane; left → (5.08, 5.60) +95° = west-arm lane; straight → (5.85,
   5.15) 0° = south lane. The bot lane-follows away after each.
5. **Other-bot detection = real robot models with vehicle plates** (the Duckietown way; YOLO has no
   robot class — classes are duckie/truck/sign). `Sign_vehicle` billboard REMOVED; a **ParkedBot**
   (NPC model) at (6.12, 4.05) carries a 0.11 m tag-400 plate (`obj_vehicle_tag.tscn`); the **moving
   NPC** carries front+back plates too (ambient encounters at ~0.2 m/s on its loop).
6. **Dashboard test UX** (`http://localhost:5000/`): a **Live bot status** panel (state, pose, signs
   with est-distance, light, **red line AT-LINE flag**, obstacle, legal turns, wheels — polls
   `/telemetry`) and **“Watch a behaviour” buttons** (`POST /scenario/{lane,stop,light,obstacle,robot}`)
   that teleport the bot to a scenario start and restart the agent so you watch it live on `/video`.
   Verified live: stop scenario → DRIVE→APPROACH→STOPPED (red_line=True @ z 6.29)→1.5 s→TURN_RIGHT→
   DRIVE eastbound on the correct lane.

**New logic gates (all evidence-driven, hardware-safe, config knobs):**
- `sign_react_distance_m 1.5` — DRIVE only reacts to stop/yield/light signs within 1.5 m (sim decodes
  tags at 3 m; reacting that early creeps forever / reacts to the NEXT junction). `est=inf` (uncalibrated
  camera) keeps react-on-sight.
- `light_arm_distance_m 1.5` — arm the light detector only for a CLOSE t-light tag. Found via telemetry:
  the bot at the 4-way stop line saw the light 3.6 m down the corridor showing red and lawfully waited
  ~6 s for green — a light governing a different junction.
- `line_straight_distance_m 0.6` — final approach creeps DEAD STRAIGHT (lane markings end at the box;
  lane-steering there yanked the heading −37°).
- APPROACH accumulates **all** signs seen (`intersection_signs` by id) so the legal-turn set knows the
  4-way sign even after it scrolls out of view at the line.

## S5.1 — Current scene truth (supersedes S4.1 layout)
- 4-way cross at (5.7, 5.7); southbound approach lane x≈5.85 from z 7.05 (runway from the col9 curve).
- `Sign_stop` (octagon+tag1) at **(6.08, 6.15)**, `Sign_4way` (tag8) at **(6.08, 6.5)**, both facing the
  approach. Red lines: N (5.85, 6.08) · S (5.55, 5.32) · E (6.08, 5.55) · W (5.32, 5.85) + light line
  (5.85, 3.1). Bot stops at pose z≈6.32 (line ROI sees the line ~0.2 m ahead of the camera → bumper at
  the line).
- Light corridor (col9 south): `Sign_tlight` (tag74) at (6.0, 3.2) · red line z 3.1 ·
  `TrafficLight_demo` at **(6.05, 2.7)** — restyled: black housing + small 0.055 lens at y 0.47 (reads
  red/yellow/green at the line and during approach).
- `ParkedBot` + plate at (6.12, 4.05). NPC loop unchanged (bottom/west roads) with plates.
- Spawn unchanged (0.9, 0.155, 5.1).

## S5.2 — How to run (POWERSHELL syntax — `VAR=x cmd` is bash-only and fails in PS!)
```powershell
# Behaviour suite (12/12; launches its own Godot; logs/frames per scenario):
.venv311\Scripts\python.exe tasks\project\sim_tests\behaviour_suite.py
# Live dashboard (needs PYTHONUTF8 for launch.py's emoji prints):
$env:PYTHONUTF8=1; .venv311\Scripts\python.exe launch.py --sim --task project
# then open http://localhost:5000/ → "Watch a behaviour" buttons + Live bot status panel
# offline logic (18/18):
.venv311\Scripts\python.exe tasks\project\sim_tests\verify_task2.py
# kill orphans first if a launch fails:
Get-Process Godot* -ErrorAction SilentlyContinue | Stop-Process -Force
```

## S5.3 — Verified (Session 5)
- ✅ behaviour_suite **12/12 × 3 consecutive runs** (random turn differed each run → all of RIGHT/LEFT/
  STRAIGHT exercised): red-line stop pose z≈6.32 heading ≈0°, pause 1.53–1.55 s, deterministic arc
  landings, light armed+stopped at its line (colors red/yellow read), duckie SOFT_STOP, parked-bot
  plate detected + `we_go_first` evaluated. ✅ verify_task2 18/18. ✅ Dashboard live run of the stop
  scenario end-to-end (telemetry sequence captured in S5.0.6).
- ⚠️ Still true: host sorts after `megabot01` → bot always goes first here (yield branch offline-tested);
  one controllable bot (parked/NPC bots are detection targets, not negotiating agents); HW needs its own
  calibration for ticks/HSV/intrinsics/stop distances (all config knobs).
- ⚠️ `turn_seconds` fallback values are stale relative to the new arcs (only used by an encoder-less
  bot; sim + real bot both use ticks).

## S5.4 — HARDWARE READINESS (configs + calibration tooling, this session)

The bot now deploys with **hardware-correct starting values** via per-platform overlays that ONLY
`real_server.py` applies (the sim keeps the base sim-verified YAMLs; suite re-proved 12/12 after):

- `agent.main` gained two more no-op-default kwargs: `timings_override` (real_server passes base
  `maneuver_timings.yaml` ⊕ `maneuver_timings_bot.yaml`) and `lane_config_path`
  (→ `config/lane_servoing_config_bot.yaml`). Same hardware-safe pattern as observer/intrinsics.
- **Why an overlay is REQUIRED:** real `DaguWheelsDriver` runs `pwm_min=60` → PWM = v·195+60, which
  **compresses wheel-speed ratios** — the sim's arc pair 0.36/0.45 would turn much wider on hardware.
  Overlay values 0.30/0.45 (right) & 0.35/0.45 (left) reproduce the sim-verified PWM ratios; ticks
  ×0.975 for real 0.585 m tiles (sim tiles are 0.6 m).
- **`config/camera_intrinsics.yaml`** (all-commented scaffold): until measured, the bot runs the safe
  *uncalibrated* mode (react-on-sight + red-line primary stop). Note the deploy tarball ships
  `tasks/<task>/packages` + `config/` + `servers/<task>` ONLY — `duckiebot/…/camera_config.yaml` never
  reaches the bot, which is why intrinsics live in `config/`.
- **`tasks/project/sim_tests/calibrate_real_bot.py`** (laptop-side, works on the bot's live `/video`
  stream): `--measure-fx --distance D` (fits fx through pupil's own pose estimate — a naive pinhole
  `side·D/size` disagrees with pupil by 25–50%, found the hard way; **validated against the live sim:
  recovered fx 260 vs the calibrated 252, ~3%**), `--verify`, and `--hsv` (red-line + light check on
  site). Full on-site procedure incl. trim + one-trial turn correction: **BOT_BEHAVIOR.md §1b**.
- modcon (`gain 1.0/trim 0`) defaults are neutral; trim is the per-bot drive-straight knob (§1b step 1).

---

# (SESSION 4 below — fidelity model, still current) Sim runs the bot's real code paths

> ⚠️ Scene/config specifics in S4 and older sections are superseded by S5 above.

---

## S4.0 — What changed and why (the fidelity rebuild)

**Problem:** the old sim could not tell us if the code works on the bot, because it ran *different code*:
(1) no camera intrinsics → `est_distance_m = inf` → `is_at_stop_line` used a **pixel proxy**, not the real
`est_distance < stop_distance` path; (2) `GodotWheelsDriver.encoders = None` → **time-based** turns, not the
real **encoder-tick** path; (3) inflated/uneven tag sizes. And nothing logged what the bot actually saw/did.

**Now (all hardware-safe — real `DaguWheelsDriver`/real-camera path is byte-for-byte preserved):**

1. **Simulated wheel encoders** (`duckiebot/wheel_driver/godot_wheels_driver.py`): `SimWheelEncoder`/
   `SimWheelEncoderPair` model the real DB21J encoder (`135 ticks/rev`, `r=0.0318 m`) by integrating the
   executed wheel command over wall-clock. So `maneuvers._await_motion` takes the **encoder branch** in sim,
   identical to HW. Turn angle is now geometry-faithful (a turn tuned to 90° in sim is ~90° on the bot).
   **Measured:** `turn_ticks=90` → only **69°** (the old value was geometrically wrong); recalibrated to
   **`turn_ticks=118` → ~90°** (`measure_turns.py`: left +92°, right −88/−92°). `straight_ticks 60→405`
   (one 0.6 m tile; 60 was only 0.089 m).
2. **Sim camera intrinsics + real est_distance** (`apriltags.py` + `agent.py`): `AprilTagDetector` gained an
   optional `intrinsics=(fx,fy,cx,cy)` param; `agent.main` gained keyword-only `observer`,
   `apriltag_intrinsics`, `apriltag_tag_size` (all default to today's exact behaviour → HW unchanged). The
   **sim** passes calibrated **`fx=fy=252, cx=320, cy=240, tag_size=0.20 m`** (in
   `tasks/project/sim_tests/sim_telemetry.py`, via `sim_fidelity_kwargs()`), so `est_distance_m` is REAL
   (verified within ~3% of ground truth by `calibrate_intrinsics.py`). `is_at_stop_line`'s pixel proxy is
   now **gated to `est_distance == inf`** so it never pre-empts the real path when intrinsics exist.
3. **Telemetry** (`sim_telemetry.py` `TelemetryLogger`): the `observer` logs, every loop, what the agent
   perceived + decided (state, event, wheels, every tag with `est_distance`, light colour, obstacle, lane
   error) enriched with Godot **pose** (via new sim-only `GodotWheelsDriver.poll_state()`), to JSONL +
   `/telemetry`. **This is the "see what the bot reacts to" tool** — it is how the bugs below were found.
4. **New logic (hardware-safe, evidence-driven):** `stop_distance_m` + `stop_commit_distance_m` are now
   config knobs; a **commit latch** in APPROACH stops the bot once a sign was identified within
   `stop_commit_distance_m` and then scrolls out of the (down-tilted) camera frame — a real bot commits to
   the stop, it doesn't drive through because the sign left the top of the frame.

## S4.1 — Current scene (`scenes/maps/project.tscn`) — THIS is current (supersedes §5/§6/§8 layout)

- **Tile size = 0.6 m**; tile `(col,row)` center = `(0.6·col+0.3, 0.6·row+0.3)`. **4-way cross at col9 =
  `(5.7,5.7)`**; **col9 is a long N–S road** (z 1.5→6.9). Bot's **southbound right lane on col9 ≈ x=5.85**.
- **Spawn unchanged: `(0.9, 0.155, 5.1)` facing −Z** — the west road (col1), a clean straight the bot
  lane-follows from t=0 (verified live).
- **Rubric signs moved onto the col9 east roadside (`x=6.0`), FACING NORTH (identity basis) so a southbound
  bot sees them head-on, with tags LOWERED to y≈0.22 m** (into the 14°-down camera's view; billboards were at
  y=0.45 and fell out the top at close range). Spaced N→S: **`Sign_stop`(tag1)@z4.5 · `Sign_4way`(tag8)@z4.0
  · `Sign_vehicle`(tag400)@z3.6 · `Sign_tlight`(tag74)@z3.2 · `TrafficLight_demo`@z3.0**. Tag planes unified
  to **0.20 m** (`obj_tag_billboard.tscn` 0.23→0.20). Octagon stop sign unchanged size.
- Sign decode is intermittent below ~0.4 m (sim double-JPEG + close-range skew), so the bot **commits to the
  stop at ~0.5 m** (`stop_distance_m=0.5`, reliably detected). On HW (cleaner camera) it should reach the
  ~0.25 m line — recalibrate `stop_distance_m` there.

## S4.2 — Config now (`packages/config/maneuver_timings.yaml`, all sim-calibrated knobs)
`turn_ticks 118` (≈90°), `straight_ticks 405` (≈0.6 m tile), `turn_inner_speed 0.05`,
`turn_outer_speed 0.45`, `straight_speed 0.3`, `base_speed 0.25`, `approach_creep_speed 0.25`,
`ramp_max_step 0.05`, `sign_cooldown 4`, **`stop_distance_m 0.5`**, **`stop_commit_distance_m 0.5`**,
`turn_seconds 0.45`/`straight_seconds 2.2` (now only a fallback for an encoder-less bot).

## S4.3 — How to run + verify (Session 4)
```bash
# ⭐ Per-behaviour evidence suite (launches Godot, runs the REAL agent w/ fidelity+telemetry):  11/11
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv311/Scripts/python.exe tasks/project/sim_tests/behaviour_suite.py
# turn calibration (encoder path):  left+92 right-88 straight 0
PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/measure_turns.py
# camera intrinsics calibration + sign-facing probe:  fx≈252, est within ~3%
PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/calibrate_intrinsics.py
# offline decision logic:  18/18
PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/verify_task2.py
# live sim + faithful agent + live telemetry:  GET http://localhost:5000/telemetry
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv311/Scripts/python.exe launch.py --sim --task project
```
Logs land in `_sim_logs/behaviour/<ts>/<label>.{jsonl,summary.json,png}` and `_sim_logs/live/live.jsonl`.
**Read the JSONL** (pose + est_distance per tag per frame) to see exactly what the bot reacted to.

## S4.4 — Verified (Session 4)
- ✅ `behaviour_suite.py` **11/11**, twice (lane-follow; stop sign DRIVE→APPROACH→STOPPED→turn via real
  est_distance; turns ±~90°; straight ~0 drift; traffic light arm→stop→proceed; duckie SOFT_STOP;
  vehicle-tag precedence). ✅ `verify_task2.py` 18/18. ✅ live `launch.py --sim` faithful + `/telemetry`.
- ⚠️ **Two-robot precedence**: still one controllable bot; a Vehicle tag (400=`megabot01`) stands in. Host
  sorts first here so the bot always goes first; the yield branch is offline-tested.
- ⚠️ **Turns are still open-loop dead-reckoning** (now geometry-correct, but no closed-loop landing). The
  suite uses teleports per behaviour; a single continuous "drive the whole loop hitting every sign" demo is
  still limited by open-loop turn landing.
- ⚠️ **HW to recalibrate** (sim gives geometry/code-path-correct *starting points*, not final HW numbers):
  `turn_ticks`/`straight_ticks` (wheel slip), `stop_distance_m` (real sign placement),
  AprilTag camera intrinsics (real IMX219), and the lane/light HSV YAMLs (real lighting).

## S4.5 — Hardware-safety audit (Session 4 changes)
- Sim encoders live only in `GodotWheelsDriver` (sim class); real bot uses `DaguWheelsDriver` → real
  `WheelEncoderPair`. `agent.main`'s new kwargs default to no-op; `real_server.py` passes none → real
  file-search intrinsics + 0.065 m tag + no observer. `is_at_stop_line` real branch unchanged; the proxy is
  only *more* restricted (inf-gated). `turn_ticks`/`straight_ticks`/`stop_distance_m` are documented
  calibration knobs. Net: the on-bot agent path is preserved and every behaviour change reads from config.

---

## 0d. (Session 3) STATUS UPDATE — signs reworked, demo integrated into the sim, full reset added

Addressed four concrete complaints:

1. **No more giant floating "QR codes".** `obj_tag_billboard.tscn` shrunk from a 0.40 m tag on a 0.56 m
   board to a normal **0.23 m tag panel on a post** (a real-Duckietown-looking sign). It's used for the
   intersection / t-light / vehicle signs.
2. **Road signs now carry their AprilTag.** `obj_stop_sign.tscn` (the red octagon) now renders its tag on a
   **0.20 m white-quiet-zone panel** below the octagon (was a 0.07 m tag on a red background → barely
   decoded). The tag-less decoy octagons that cluttered the map (and occluded the real stop sign) were
   **removed**. Every sign in `project.tscn` is now a tagged, normal-sized sign.
   - Sizing note: with no camera intrinsics the sim needs ~0.2 m tags to decode at ~1 m. To let the bot
     stop while it can still see lane markings, `is_at_stop_line`'s **sim-only** px proxy was lowered
     `60→38 px` / `500→200` lane px. Real hardware uses the `est_distance_m < 0.25` branch — unchanged.
3. **The task demo runs INSIDE the live sim now.** The dashboard has a **"▶ Run task demo"** button
   (`/demo` + `/demo_status`) that teleports the bot through every rubric scenario and runs the REAL agent
   in the SAME Godot window, streaming PASS/FAIL to the page. Verified live: stop-sign
   `DRIVE→APPROACH→STOPPED→TURN→DRIVE`, light `…→STOPPED→WAIT→STOPPED→TURN→DRIVE` (stop-red/go-green),
   obstacle SOFT_STOP, precedence (detect `megabot01` + `we_go_first`). (`demo_tasks.py` still exists as a
   standalone/CI version — same scenarios, launches its own Godot.)
4. **"Restart simulation" button** (`/reset`): teleports the bot back to spawn (Godot `reset_game`) AND
   restarts the agent — a full reset, not just the agent-thread restart that "Restart agent" does. Verified.

Sign-decode + all behaviors re-verified: **demo_tasks.py 9/9** and the dashboard `/demo` all green.

---

## 0c. STATUS UPDATE — Tasks 2 & 3 are now DEMONSTRATED (demo_tasks.py 9/9)

`tasks/project/sim_tests/demo_tasks.py` is the authoritative "Tasks 2 & 3 are complete" proof. It uses a
new sim **teleport** hook (`Moveee.gd.teleport` + `WheelCommandServer` "teleport" msg + `GodotWheelsDriver.
teleport`, all sim-only) to place the bot facing each rubric situation, then runs the REAL `agent.main`
and checks the reaction. **Latest run: 9/9 PASS** incl. the full light cycle
`see_light→APPROACH→STOPPED→WAIT(red)→STOPPED→go(green)` and stop-sign→stop→random-turn. See SIM_TESTS.md
for the rubric→scenario→evidence matrix. Run it:
`PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/demo_tasks.py`.

Scene note: the autonomous spawn is now `(0.9, 0.155, 5.1)` (LEFT straight) — drives ~18 s of clean
on-road loop before it reaches the col-9 traffic light. The light + `t-light` tag live on col-9-south
(`light @6.05,3.0`, `Sign_tlight_demo @5.5,3.0`) — deliberately in an octagon-free stretch so the HSV
detector sees ONLY the light (no red stop-sign octagons to confuse it); that's why `demo_tasks` teleports
there for a CLEAN light demo. The autonomous bot stops correctly at that light; the post-green random turn
on a straight is the known dead-reckon limit (see §0b) — for a clean autonomous turn it'd need a real
intersection or closed-loop turns. demo_tasks (teleport) sidesteps this and shows each behaviour cleanly.

---

## 0b. STATUS UPDATE — "drives off road at spawn" is FIXED (verified in full sim)

After the spin fix below, the user reported the bot still **turned and drove off-road at spawn**. Root
cause (finally SEEN by capturing frames from t=0 with `spawn_capture.py` — the spawn moment every prior
harness missed): the bot spawned **one tile north of the 4-way cross with a stop sign right there**, so it
stopped and did a **blind dead-reckoned turn before lane-following ever stabilized** → into the sign post /
off the lane. A second trap: the `t-light-ahead` billboard sat **mid-straight**, triggering a stop+turn
where there is no junction → straight off-road.

**Fixes:**
- **Spawn moved** to `(5.832, 0.155, 5.1)` — south of the cross, onto a 4-tile straight with **no
  turn-triggering signs ahead**. The bot now lane-follows from t=0.
- **`Sign_tlight_demo` moved off the straight** to `(5.35, 0, 6.05)` (by the cross). RULE LEARNED:
  **turn-triggering tags (stop/yield/t-light) must ONLY sit at real intersections** — mid-straight they
  turn the bot off-road. (4-way / vehicle tags are safe mid-straight; they don't trigger a turn.)

**Verified (full `launch.py --sim`, dashboard + 2 YOLOs, and `spawn_capture.py` frames):** the bot drives
a **full ~75 s loop staying on-road** — straights, curves, and intersection pass-throughs — in continuous
DRIVE, with only correct brief duckie SOFT_STOPs. No spin, no off-road. **This is the headline fix.**

⚠️ **KNOWN REMAINING LIMITATION (be honest with the user):** intersection **turns are blind dead-reckoning**
(no localization — a fixed ~90° rotation). They land roughly on a lane at a real 4-way but are NOT robust,
and they send the bot off-road if triggered where the geometry doesn't match. Because of this, the current
demo route (south loop) **drives well but does not exercise stop-sign turns or the traffic light** (the bot
doesn't face those tags on this loop). Making Task-2/3 demos reliable needs EITHER per-intersection scene
choreography (place + face + test each trigger at a real junction) OR closed-loop turns using bot position
(telemetry now available via `get_state` heading/pos — see §4). This is the next real work item.

---

## 0. STATUS — the "360 spin" is root-caused, fixed, and MEASURED

**Root cause (definitive, measured):** intersection turns over-rotated by **1.4–2.9 full revolutions**.
`maneuvers.py` reused ONE number, `turn_ticks`, for two incompatible things:
- REAL robot: `turn_ticks` = outer-wheel **ENCODER ticks** (a physical ~90° rotation).
- SIM: the old code reused it as **time** (`turn_ticks × 0.05 s`). With the team's `turn_ticks=90`,
  that's `4.5 s`. The Godot bot turns at **omega ≈ (outer−inner)/baseline × max_speed = (0.45−0.05)/0.10 ×1.0
  = 4 rad/s**, so `4.5 s × 4 rad/s = 18 rad ≈ 1030° ≈ 2.9 spins` per "turn". (My earlier `turn_ticks=45`
  still gave ~515° ≈ 1.4 spins — why the user still saw spinning.)

**Why I missed it before:** I verified turn *duration* and eyeballed sparse camera frames. A 515° turn
**aliases** to ~155° net heading in a still frame, so it looked like a ~90° turn. Lesson baked into the
new harness below: measure rotation by **integrating heading at ~50 Hz**, never from before/after stills.

**The fix:** split the two meanings. `maneuvers._await_motion` uses `turn_ticks` (encoders) on hardware and
a NEW `turn_seconds` in sim. Measured result (heading-integrated, `measure_turns.py`):
`turn_left +92°, turn_right −99°, straight 0° drift`. Verified end-to-end in the FULL sim (`launch.py --sim`,
dashboard + 2 YOLOs): approach → stop at line → clean ~90° turn → resume centered. **No spin.**

---

## 1. CRITICAL environment gotchas (unchanged, still true)

1. **Use `.venv311`** (Python 3.11): `C:/MyProjects/FinalSoftware/.venv311/Scripts/python.exe`.
   Default `.venv` is 3.14 with **no `pupil_apriltags` wheel** → AprilTags silently disabled (bot blind to
   all signs/lights but still lane-follows).
2. **Prefix launches with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`** (else `launch.py` emoji crashes the
   Windows cp1252 console).
3. **Free ports 5000/5001/5002 before each launch** (kill `Godot*` + any `python` running `launch.py`/
   `sim_tests`). An orphan holding the camera port makes the sim "fail to start".
4. Use absolute paths; `_sim_logs/` is gitignored (mkdir if missing).
5. Godot is a Windows **GUI app** → its `print()` does NOT reach the launch log. Observe via `/video`,
   frame captures, or the new heading telemetry (§4) — not Godot stdout.

## 2. How to run
```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv311/Scripts/python.exe launch.py --sim --task project
# dashboard http://localhost:5000/  (camera + 2x2 debug grid + live config + Drive control)
```
Real robot (same agent, different drivers):
`python launch.py --run --bot <name> --task project`  /  `--stop --bot <name>`.

---

## 3. CHANGES THIS SESSION — and why each is HARDWARE-SAFE

`real_server.py` runs the SAME `agent.main(camera, wheels, leds, stop_event)` as the sim — only the driver
objects differ (`DaguWheelsDriver`/`CameraDriver`/`LEDDriver` vs Godot ones). So every logic change runs on
the bot too. Audited:

| Change | File | Hardware effect |
|---|---|---|
| **Split turn timing**: `_await_motion(... ticks, seconds ...)`; encoders→ticks, sim→seconds | `maneuvers.py` | **None.** Real `DaguWheelsDriver.encoders = WheelEncoderPair()` (wheels_driver.py:48) → encoder branch → `turn_ticks=90` as before. `turn_seconds` only read when `encoders is None` (sim, or a bot whose GPIO encoders failed to init). |
| `turn_ticks` restored **45→90**; added `turn_seconds: 0.45`, `straight_seconds: 2.2` | `maneuver_timings.yaml` | Real turn = team's 90 encoder ticks (unchanged). Sim turn = 0.45 s ≈ 92–99°. |
| **Reverted SOFT_STOP timeout** → wait until obstacle clears | `agent.py` | **Safer + correct.** The old timeout drove the bot THROUGH the duckie after 3 s = pedestrian hit on hardware, game-over in sim. Now matches README ("soft-stops until the duckie is removed"). |
| Removed `ignore_obstacles` machinery (obstacle check always on) | `agent.py` | Always brakes for obstacles. |
| **Bounded, gentle lane recovery** (2-phase: brief gentle sweep toward last-seen lane, then straight) | `visual_lane_servoing/agent.py` | Lane-loss no longer freezes (old dead-stop) NOR spins (a tight pivot). On hardware: slow gentle re-acquire, never a violent spin. `recovery_speed 0.15`, `recovery_turn 0.08`, `recovery_max_frames 12`. |
| Kept from earlier: `see_light`→APPROACH, `_clear_to_enter` (stop-red/go-green/hold-yellow/yield/obstacle), APPROACH **creep** to line, `sign_cooldown` after a turn | `states.py`, `agent.py`, `maneuver_timings.yaml` | All correct on hardware (roll up to the stop line; don't re-trigger the just-obeyed sign; obey lights). |
| Lane gains `p_gain 0.45, d_gain 0.15`, error filter 0.5/0.5 | `lane_servoing_config.yaml` | Sim-tuned; deploys to bot but is a **calibration knob** (README §6). Bot needs its own tuning; the code reads from config so it's tunable live in the dashboard. |
| **Sim telemetry**: bot heading/pos in Godot `get_state`; parsed into `GameState` | `Moveee.gd`, `godot_wheels_driver.py` | **Zero hardware impact** — both are sim-only (real robot uses `DaguWheelsDriver` + real camera, never this). |

**Net hardware story:** the real-robot agent path is preserved (encoder turns, always-stop-for-obstacles,
wait-for-pedestrian). The only on-hardware behavior *changes* are improvements that read from config:
creep-to-stop-line, bounded lane recovery, and the Task-2 traffic-light logic.

⚠️ **Hardware to-calibrate before a real run:** `turn_ticks`/`straight_ticks` (encoder ticks for ~90°/one
tile — team's job on the bot), lane `p_gain`/`d_gain` (real camera/lighting differ), `traffic_light_hsv.yaml`
(real light colors), and `lane_servoing_hsv_config.yaml` (real lane colors). The sim values are NOT expected
to be right on hardware — they're tunable knobs, not logic.

---

## 4. Verification harnesses (`tasks/project/sim_tests/`, run under `.venv311`, ports free)

- **`measure_turns.py`** ⭐ NEW — launches Godot alone, runs REAL `maneuvers.turn_left/right/straight`,
  **integrates Godot heading at ~50 Hz** to report actual degrees. THE tool that caught the multi-spin.
  Last result: `turn_left +92°, turn_right −99°, straight 0° drift → ALL TURNS SANE`.
- **`spawn_capture.py`** ⭐ NEW — Godot alone + REAL `agent.main`, saves a camera frame every 0.2s **from
  t=0** + logs transitions. THE tool that finally showed the spawn behavior (Read the PNGs in
  `_sim_logs/spawn/`). Use this to see what the bot does in the first seconds — the part `/video` grabs miss
  (the dashboard isn't up yet at spawn).
- **`agent_trace.py`** — Godot alone + REAL `agent.main`, logs every state transition + wheel command.
  Caveat: NOT the full sim (no dashboard / 2nd YOLO). Good for FSM logic; pair with a full-sim frame grab.
- **`verify_task2.py`** — offline deterministic FSM/entry-gate/precedence checks (18, all pass).
- **`verify_perception.py`** — needs the full sim up; reports tags + light colors seen, saves frames.
- **Heading telemetry**: `Moveee.gd get_state` now returns `heading_deg`, `pos_x`, `pos_z`; parsed into
  `GodotWheelsDriver.transport.game_state`. Use it to MEASURE rotation / map the route — do NOT judge
  rotation from sparse camera stills (they alias: 515° looks like 155°).
- (Temp scripts `_diag.py`/`_spin_diag.py`/`_grab.py` were used and removed. To re-observe the full sim,
  grab `http://localhost:5000/video` MJPEG frames and Read the PNGs.)

---

## 5. Current scene (`GodotSimulation/ducky-bot/scenes/maps/project.tscn`)

- **`DuckieBot` at (5.832, 0.155, 5.1)** facing −Z — SOUTH of the 4-way cross, on a clean straight
  (tiles 9_8→9_5, z=5.1→3.3) then a curve at 9_4 (z=2.7). Drives from t=0 with no immediate maneuver.
  3 intersections total: 4-way @(5.7,5.7), T @(2.1,3.3), T @(6.9,2.7).
- White tag billboards (45°, roadside-left), all `obj_tag_billboard.tscn` (`sign_texture` = a tag36h11 PNG):
  `Sign_stop_demo` (tag 1 stop) @(5.35,5.9, by the cross) · `Sign_4way_demo` (tag 8) @(5.35,4.6) ·
  `Sign_vehicle` (tag 400=megabot01) @(5.35,3.4) · `Sign_tlight_demo` (tag 74) @(5.35,6.05, by the cross —
  MOVED off the spawn straight). NOTE: on the south spawn loop the bot does NOT face the stop/t-light tags,
  so it just drives (see §0b limitation).
- `TrafficLight_demo` cycling lens (red 7s/green 6s/yellow 2.5s) @(6.05,3.0).
- Red-octagon decoy models `Sign_11_5`/`Sign_11_4` moved to (2.4,4.2)/(2.4,4.8) — away from the light so the
  HSV light detector doesn't see red octagons next to the light.
- 12 duckies under `Ducks/` (roadside; flicker into YOLO occasionally → brief correct SOFT_STOP).

## 6. Config (current; all live-tunable in the dashboard)
- `maneuver_timings.yaml`: approach_creep_speed 0.12, base_speed 0.25, ramp_max_step 0.05, sign_cooldown 4.0,
  straight_seconds 2.2, straight_speed 0.3, straight_ticks 60, turn_inner_speed 0.05, turn_outer_speed 0.45,
  **turn_seconds 0.45, turn_ticks 90**.
- `lane_servoing_config.yaml`: base_speed 0.25, p_gain 0.45, d_gain 0.15, max_steer 0.55,
  detection_threshold 100, recovery_speed 0.15, recovery_turn 0.08, recovery_max_frames 12, curve_* (unused —
  `detect_curve` is a no-op stub).

## 7. What's verified vs. open
- ✅ Turns are ~90° (measured), no spin — sim, full `launch.py --sim`.
- ✅ Stop-sign cycle `DRIVE→APPROACH→STOPPED→TURN→DRIVE`, no re-trigger loop (`agent_trace`).
- ✅ Task-2 light logic (offline 18/18). Light end-to-end (`see_light→STOPPED→WAIT on red→go on green`) was
  observed in an earlier session; ⚠️ **re-confirm on the current spawn/route** (random turns may not pass the
  light every lap — drive to it via dashboard manual mode if needed).
- ⚠️ Two-robot precedence: only a Vehicle TAG is present (logic runs, `we_go_first` tested). No 2nd moving
  bot. Host=`Toma_PC` so `megabot01` never has precedence here → bot always goes first (won't deadlock).
- ⚠️ Hardware: logic is hardware-safe (§3) but turn/lane/HSV values need on-bot calibration before a real run.

## 8. Reference facts
- AprilTag DB (`apriltagsDB.yaml`): 1=stop, 2=yield, 8=4-way, 9=right-T, 10=left-T, 11=T-int,
  74–94 & 200–230=t-light-ahead, 95–124=duck-crossing, 125–131=parking, 400+=Vehicle(megabotNN).
  Tag textures: `textures/tag36h11/tag36_11_000NN.png`.
- Sim physics (`Moveee.gd`): `omega = (right−left)/baseline × max_speed = (right−left)×10 rad/s`, clamp ±8;
  `max_speed=1.0 m/s`. This is why turn duration maps directly to angle in sim.
- Real `DaguWheelsDriver` HAS encoders → maneuvers use the **encoder (tick) path** on hardware.
- Agent FSM (`states.py`): DRIVE→(see_stop_or_yield|see_light)→APPROACH→(at_stop_line)→STOPPED→
  {TURN_*|STRAIGHT_THROUGH}→DRIVE; STOPPED→WAIT (red light / yield); DRIVE/APPROACH+obstacle→SOFT_STOP→
  (cleared)→DRIVE.
