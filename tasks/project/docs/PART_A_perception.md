# Part A — Perception ("See")

**Owner:** Person A
**Job:** turn raw camera frames into *facts* — what tags are visible and what they
mean, what obstacles/other-bots are ahead, and whether the bot is at a stop line.
**No decisions, no motors.** Part B (Decide) and Part C (Act) consume what this layer
produces.

> The project no longer uses traffic lights. `perception/traffic_light.py` and
> `config/traffic_light_hsv.yaml` remain in the tree and still function, but are
> **out of scope** and belong to no part — see the note at the end.

## Files owned

| File | Role |
|---|---|
| `packages/perception/apriltags.py` | `AprilTagDetector` + `TagObservation` — frame → tag IDs (+ distance) |
| `packages/sign_registry.py` | `SignSemantic` + `lookup()` — tag ID → meaning (loads `apriltagsDB.yaml`) |
| `packages/perception/duck_hsv.py` | `detect_duckies_hsv`, `detect_vehicles_hsv` — HSV obstacle / other-bot detection |
| `packages/perception/intersection.py` *(detection half)* | `is_at_stop_line`, `red_line_pixel_count`, `red_line_threshold`, `detect_red_line`, `red_band_metrics` |

> `intersection.py` is **shared**: the "am I at the line?" detection above is Part A;
> the "which turns are legal?" logic (`merge_turn_constraints`) is **Part B**.

---

## 1. `apriltags.py` — detecting tags

### `TagObservation` (dataclass)
```python
id: int                 # the tag36h11 ID
center_xy: (int, int)   # pixel centre
side_length_px: int     # mean of the 4 corner edge lengths — the proximity proxy
est_distance_m: float   # metric distance, or inf when uncalibrated
est_yaw_rad: float      # tag yaw, or 0.0 when uncalibrated
```

### `AprilTagDetector(tag_size_m=0.065, intrinsics=None)`
- **Backend auto-select:** tries `pupil_apriltags` then `dt_apriltags` (same API),
  family `tag36h11`. If neither imports it degrades to "no tags" (returns `[]`) instead
  of crashing — the bot still drives, just blind to signs. Backend is printed once.
- **Intrinsics:** uses a passed `(fx,fy,cx,cy)` if given (the sim does this); else
  `_try_load_intrinsics()` searches `duckiebot/camera_driver/config/camera_config.yaml`
  then `config/camera_intrinsics.yaml`. None found → uncalibrated.

### `detect(bgr_frame) -> list[TagObservation]`
1. Empty frame / no backend → `[]`.
2. `gray = cvtColor(bgr, BGR2GRAY)`.
3. **Calibrated** (intrinsics): `detect(gray, estimate_tag_pose=True, camera_params, tag_size)`
   → `est_distance_m = ‖pose_t‖`, `est_yaw_rad = atan2(R[0,2], R[2,2])`.
4. **Uncalibrated:** plain `detect(gray)` → `est_distance_m = inf`, yaw `0.0`, and a
   one-time "no intrinsics" warning.
5. `side_length_px` = mean of the four corner edge lengths (grows as the tag nears).

> **Two camera modes drive the whole system.** *Calibrated* → real metres, so reactions
> are distance-gated. *Uncalibrated* (the bot's current default) → `inf` distance, so
> downstream logic falls back to **tag pixel size** for "how close."

---

## 2. `sign_registry.py` — tag ID → meaning

- On import, loads `apriltagsDB.yaml` **once** (552 entries) into
  `_REGISTRY: dict[int, SignSemantic]`. A failed load logs and leaves it empty — never
  crashes import.

```python
@dataclass(frozen=True)
class SignSemantic:
    kind: str               # 'stop', '4-way-intersect', 't-light-ahead', '' (none) ...
    tag_type: str           # 'TrafficSign' | 'Localization' | 'Vehicle'
    vehicle_name: str|None  # 'megabot01' etc. (Vehicle tags only)
    @property
    def available_turns(self) -> set[str]
```

`available_turns` mapping (`_TURN_MAP`): `4-way-intersect`→{l,r,s}, `T-intersection`→{l,r},
`right-T-intersect`→{s,r}, `left-T-intersect`→{s,l}, `oneway-right`→{r}, `oneway-left`→{l},
`do-not-enter`→{}, anything else → {l,r,s}.

`lookup(tag_id)` → `SignSemantic` or `None`. Key IDs: `1`=stop, `8`=4-way, `9`=right-T,
`10`=left-T, `11`=T-int, `400+`=Vehicle (`megabot01…`).

> `available_turns` is the *data*; the *combining* of multiple signs into a legal set
> happens in Part B (`merge_turn_constraints`).

---

## 3. `duck_hsv.py` — HSV obstacle & other-bot detection

Exists because the trained YOLO model can't load on the bot (Jetson, OpenCV 4.1.1, no
onnxruntime). Dependency-free, and returns the **same format** as the YOLO detector so it
drops into the obstacle path unchanged.

### `detect_duckies_hsv(bgr, cfg=None, lane_xs=None) -> [((x1,y1,x2,y2), score, 0)]`
The hard part: the yellow lane centre line is the *same colour* as a duck. So after the
yellow HSV mask + morphology, blobs are rejected by **shape/position**:
- `min_height_frac` — a duck close enough to brake for is tall; dashes are short.
- aspect (`min/max_aspect`) — a standing duck is taller-than-wide; dashes are wider.
- `max_solidity` — a painted dash fills its rotated rect (~1.0); a rounded duck doesn't.
- `reject_bottom_square` — a chunky near-dash running off the bottom edge.
- optional lane-line exclusion via `lane_xs` (a blob *on* the yellow line is a dash).
`score` = bbox area / `score_area_norm` (a proximity proxy), class `0`.

### `detect_vehicles_hsv(bgr, cfg=None) -> [((x1,y1,x2,y2), score, 1)]`
A big **blue** blob, low and centred (a Duckiebot body ahead in the down-tilted view).
Gates: top-ignore, min area, `min_cy_frac` (low in frame), centre band, `min_fill`,
aspect. Class `1` (other-bot). This is how the bot brakes for robots that carry no tag.

---

## 4. `intersection.py` — stop-line detection (the Part-A half)

Duckietown paints a red line across the lane at junctions; with the camera pitched ~14°
down, red in the bottom rows means "AT the line." Red HSV is config-driven.

| Function | Returns |
|---|---|
| `red_line_pixel_count(bgr, cfg)` | raw red-px count in the bottom `line_roi_frac` (0.22) of the frame (two red hue ranges). Surfaced in telemetry for tuning. |
| `red_line_threshold(cfg)` | the count meaning "at the line" (`line_min_px`, 400). |
| `detect_red_line(bgr, cfg)` | `count >= threshold`. |
| `red_band_metrics(bgr, cfg)` | **structural** detector: tighter red over the central band; counts rows "on the bar"; `present` iff `band_min_rows_frac ≤ rows_frac ≤ band_max_rows_frac`. Returns `(present, diag)`. Rejects scattered red/orange noise the raw count would false-trigger on. |
| `is_at_stop_line(tag_obs, lane_mask, stop_distance_m=0.25)` | `est_distance < stop_distance` (calibrated); or, **only when distance is `inf`**, `side_length_px > 38` AND bottom-third of `lane_mask` has > 200 px (uncalibrated proxy). |

The raw count is eager (good on a clean line); the **band** is robust (used on the bot).

---

## What Part A outputs (the interface to B and C)

```python
tag_obs   = AprilTagDetector().detect(bgr)              # list[TagObservation]
signs     = [(o, lookup(o.id)) for o in tag_obs if lookup(o.id)]   # (obs, SignSemantic)
ducks     = detect_duckies_hsv(bgr)                    # [(bbox, score, 0)]
bots      = detect_vehicles_hsv(bgr)                   # [(bbox, score, 1)]
red_line  = detect_red_line(bgr, cfg)                  # bool
red_band, diag = red_band_metrics(bgr, cfg)            # bool, dict
at_line   = is_at_stop_line(tag_obs[i], lane_mask)     # bool
```

Part C's agent loop calls all of these each frame; Part B turns the results into events
and decisions.

---

## Acceptance checks (all pass)
`lookup(1).kind=='stop'` · `lookup(8).available_turns=={l,r,s}` · `lookup(74).kind=='t-light-ahead'`
· `lookup(400)=Vehicle/megabot01` · `lookup(99999) is None` · `AprilTagDetector().detect(blank)==[]`.

## Out of scope — traffic lights (removed)
`perception/traffic_light.py` (`TrafficLightDetector`, `should_brake_for_yellow`) and
`config/traffic_light_hsv.yaml` are still in the repo and importable, and the agent's
light path still works, but the team dropped traffic lights from scope — so this module
is **owned by no part** and isn't documented here. Delete it later if you want the tree
fully clean (it's woven into `agent.py`, so that's a small refactor).
