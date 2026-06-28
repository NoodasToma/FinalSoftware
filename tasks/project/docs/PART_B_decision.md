# Part B — Decision ("Decide")

**Owner:** Person B
**Job:** given the facts from Part A (See), decide *what is legal and what to do* — the
state-machine skeleton, which turns are allowed, who goes first, and the bot's full
sign-behaviour FSM. **No raw pixel work, no motor writes** — it consumes detections and
emits states/choices that Part C (Act) executes.

## Files owned

| File | Role |
|---|---|
| `packages/states.py` | `State` enum + `TRANSITIONS` table + `next_state()` |
| `packages/perception/intersection.py` *(decision half)* | `merge_turn_constraints` — legal-turn set |
| `packages/precedence.py` | `we_go_first` — right-of-way |
| `packages/sign_detection/` *(whole package)* | the bot's self-contained sign-behaviour FSM |

> `intersection.py`'s detection functions (red line, `is_at_stop_line`) are **Part A**;
> only `merge_turn_constraints` is Part B.

---

## 1. `states.py` — the finite state machine (used by the sim agent, Part C)

Eight states: `DRIVE  APPROACH  STOPPED  WAIT  TURN_LEFT  TURN_RIGHT  STRAIGHT_THROUGH  SOFT_STOP`.

`TRANSITIONS: dict[(State, event) -> State]`:

| From | Event | To |
|---|---|---|
| DRIVE | `see_stop_or_yield` / `see_light` / `see_intersection` | APPROACH |
| DRIVE | `obstacle` | SOFT_STOP |
| APPROACH | `at_stop_line` | STOPPED |
| APPROACH | `obstacle` | SOFT_STOP |
| STOPPED | `choose_turn_left` / `choose_turn_right` / `choose_straight` | TURN_* / STRAIGHT_THROUGH |
| STOPPED | `wait` | WAIT |
| WAIT | `cleared` | STOPPED |
| TURN_* / STRAIGHT_THROUGH | `turn_done` | DRIVE |
| SOFT_STOP | `obstacle_cleared` | DRIVE |

```python
def next_state(s, event):
    return TRANSITIONS.get((s, event), s)   # unknown (state,event) → stay put
```

Pure data + one lookup. Lifecycle: `DRIVE → APPROACH → STOPPED → {TURN_*/STRAIGHT} →
DRIVE`, with `WAIT` (yield) and `SOFT_STOP` (duckie) branching off.

---

## 2. `intersection.merge_turn_constraints(observed_signs) -> set[str]`

Combine every sign seen on the approach into the set of turns the bot may pick from:
- start with `{left, right, straight}`;
- for each sign whose `kind` is a constraint kind (4-way / T / right-T / left-T /
  oneway-* / do-not-enter), **intersect** with its `available_turns` (from Part A's
  `SignSemantic`);
- `no-left-turn` → discard `left`; `no-right-turn` → discard `right`.

E.g. `[T-intersection, no-right-turn]` → `{left, right}` minus right → `{left}`.
The sim agent (Part C) accumulates signs across the whole approach, then calls this at
STOPPED to get the legal set, and picks a **random** member.

---

## 3. `precedence.py` — `we_go_first(my_name, recent_vehicle_signs) -> bool`

```python
for sign in recent_vehicle_signs:
    if sign.tag_type != 'Vehicle' or sign.vehicle_name is None:
        continue
    if sign.vehicle_name < my_name:    # someone sorts before us
        return False                   # → we yield
return True
```

Pure lexicographic name ordering — no network. Empty list / all-greater names → `True`
(we go). `my_name` is the bot's hostname; the other robot's name comes from its `Vehicle`
tag via Part A's `lookup()`.

---

## 4. `sign_detection/` — the bot's sign-behaviour FSM (what the real robot decides with)

The real bot runs the alternate agent (`agent_signs`, Part C), whose brain is this
**self-contained** package. It bundles its own lightweight perception (so it doesn't
depend on Part A's detectors) and a frame-counted FSM. Files:

### `sign_behavior_config.py` — IDs, turn semantics, tunables
- `TagID` enum + `_TAG_ID_MAP` (raw tag → role) + `resolve_tag()`.
- `_TAG_TURNS`: `FOUR_WAY`→[forward,left,right], `9`→[forward,right], `10`→[forward,left],
  `11`→[left,right] (decoded correctly; the reference's 10↔11 swap is **off** here).
- `State` enum (the FSM's own): `MOVING, APPROACHING, SLOWING, STOPPED, CHECKPATH,
  POST_STOP, INTERSECT, PRE_TURN, TURNING, EXITING`.
- `SignBehaviorConfig` — every threshold/frame-count/speed as a kwarg-overridable field
  (red-line HSV + ROI, stop-hold frames, per-turn intersect frames/speeds, exit, etc.).
  Overridable live via the dashboard (`sign_config:` in the bot timings).

### `april_tag.py` — tag detection + multi-frame confirm
- `detect_tags()`: fast path uses `cv2.aruco` (DICT_APRILTAG_36h11) when opencv-contrib
  is present (sim); **raw fallback** (`_raw_detect`) needs only base OpenCV (the bot):
  adaptive-threshold → quad contours → perspective-warp to 80×80 → sample the 8×8 cell
  grid → match the lower-36-bit code against `_CODES_36H11` over all 4 rotations.
- `confirm_tags()`: a tag must be seen `tag_confirm_frames` (2) consecutive frames before
  it counts — debounces single-frame mis-reads.

### `red_line_detection.py` — `detect_red_line(fsm, frame_rgb)`
Scans a bottom strip (`red_strip_frac`), masks two red HSV ranges + morphology, then
requires a **wide horizontal** contour (aspect ≥ 1.6, width ≥ `red_min_width_frac`) low
in the strip and a min pixel ratio. That structural test (not a raw count) is what makes
it the trigger: a real stop line is a wide bar, not scattered red.

### `vehicle.py` — `vehicle_detected(detections, img_width)`
Minimal CHECKPATH helper: returns `(seen, offset)` for the first class-1 (other-bot)
detection big enough (area ≥ 800), where `offset = |centre_x − img_centre| / width`. The
detections are fed in by Part C (from Part A's `detect_vehicles_hsv` / YOLO).

### `sign_behavior.py` — `SignBehaviorFSM` (the core)
`step(frame_rgb, base_left, base_right, detections)` runs once per frame:
1. detect + confirm tags; **save** an intersection/stop/yield tag (with a 4 s expiry).
2. read the red line (suppressed while `_red_line_locked` or during the post-action
   `ignore_red` window).
3. run `_fsm_step`, which **overrides** the incoming lane command `(base_left, base_right)`
   only while handling a sign; otherwise passes the lane command straight through.

The flow when a saved sign meets the red line:
- **stop/yield** → `SLOWING` (ramp speed down) → `STOPPED` (hold `stop_hold_frames`) →
  `CHECKPATH` (sweep left/right looking for cross traffic via `vehicle_detected`; yield to
  a vehicle on the right) → `POST_STOP` → resume.
- **intersection** (4-way / T) → pick a legal turn from `_TAG_TURNS` →
  `APPROACHING` → `INTERSECT` → `PRE_TURN` (forward creep) → `TURNING` (fixed wheel speeds
  for a fixed number of frames — **open-loop**, so the sign decision beats lane-following)
  → `EXITING` → resume.
Time is frame-counted (`FPS=24`, measured via wall-clock `dt`).

### `agent_with_signs.py` — `LaneServoingAgentWithSigns`
Subclasses the reused `LaneServoingAgent`; `compute_commands(rgb, detections)` computes
the normal lane command, runs it through `SignBehaviorFSM.step`, and returns
`(left, right, sign_state)`. This is the object Part C's `agent_signs` drives.

---

## How Part B fits

- The **sim agent** (`agent.py`, Part C) uses `states.py` + `merge_turn_constraints` +
  `precedence.we_go_first` directly.
- The **bot agent** (`agent_signs.py`, Part C) uses the `sign_detection/` FSM (which has
  its own tag/red-line detection and turn logic baked in).

Either way, Part B never touches the motors — it returns the next state / wheel
*intent*; Part C actuates it.

## Acceptance checks (all pass)
FSM walk `DRIVE→APPROACH→STOPPED→TURN_LEFT→DRIVE`; unknown event stays put ·
`merge_turn_constraints([lookup(11), lookup(3)])=={left}` ·
`we_go_first('megabot02',[lookup(400)]) is False` · `we_go_first('megabot01',[]) is True`.
