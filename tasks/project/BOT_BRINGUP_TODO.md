# 🤖 Bot bring-up — open issues & solutions (execute when the bot is free)

Bot: **franky** (Jetson, **Python 3.6**). IP changes with network — currently on an
iPhone hotspot `172.20.10.x` (was `10.181.26.26`). Use the current IP in commands below.

**Confirmed working on the bot:** lane-following, camera (post-reboot), the full agent
under Python 3.6 (ported). The items below are what's left.

---

## 0. ⭐ ROOT CAUSE of "drives a bit then freezes": YOLO TensorRT build runs the Nano OUT OF MEMORY

**Proven (from the task log):**
`[TRT] [W] Tactic Device request: 136MB Available: 125MB. Device memory is insufficient`
The object detector auto-builds a **TensorRT engine** on the Jetson; the Nano doesn't have
enough device memory, so the build **thrashes for minutes and freezes the whole bot** (agent
stops mid-lane; loop timestamp frozen; a watchdog then kills it; killing it leaves the camera
locked → next start fails → reboot needed). This — NOT the camera or `/video` — is the freeze.

**FIX — implemented (in the deploy bundle), pending a clean verify:**
1. **Force CPU object detection** on the bot: `OBJDET_CPU=1` (set in `servers/project/real_server.py`)
   → `ObjectDetectionAgent` skips the TRT path + uses `CPUExecutionProvider`. No TRT build,
   no memory thrash, no freeze. Slower per-inference, but fine because of #2.
2. **Object detection runs in a throttled background thread** (`tasks/project/packages/agent.py`,
   `obstacle_detect_period_s=0.35` ≈ 3 Hz) so it never blocks the drive loop — the bot
   lane-follows immediately and obstacle braking comes online when the first inference returns.

**Why this also fixes the camera-lock cycle:** with no freeze, the process shuts down cleanly on
`--stop` and releases the camera, so restarts stop re-locking it.

**TO VERIFY (needs ONE clean boot — see procedure):** after a reboot, `--run` once; the loop
should populate `/telemetry` within ~20 s and **keep advancing** (no freeze), driving on CPU.

---

## 0b. ⭐ "Moving but going OFF-ROAD" — two causes, both fixed in code (sim-verified 12/12)

After the freeze fix the bot drove but wandered off the road. Two reasons:

1. **Control loop starved by inline detectors.** AprilTag detection (pupil_apriltags)
   AND YOLO ran INLINE in the drive loop — each ~100-300 ms on the Nano — so the bot
   steered only a few times/sec and overshot curves. **FIX:** both detectors now run in
   ONE throttled **background thread** (`tasks/project/packages/agent.py`, `detect_period_s
   ≈ 0.2`); the main loop is just camera + lane + HSV + FSM → fast, smooth steering. Sign/
   obstacle reaction lags <~0.3 s (fine; the red line is the inline primary stop trigger).
2. **Lane commands applied at HALF magnitude.** The working `visual_lane_servoing` server
   applies `lane.compute_commands` DIRECTLY (factor 1.0). The project agent did
   `wheels = lane_out * base_speed * 2`, and `base_speed=0.25` → factor **0.5**, halving
   every command → on the real motors (`pwm_min=60` stiction) the wheels sat near the
   deadzone and moved erratically. **FIX:** bot `base_speed=0.5` (`maneuver_timings_bot.yaml`)
   → factor **1.0 pass-through**, identical to visual_lane_servoing; `max_steer 0.45→0.55`
   for curve authority. (Sim has no stiction so it never showed this.)

⚠️ **Config note:** the project task uses `lane_servoing_config_bot.yaml` + the shared
`lane_servoing_hsv_config.yaml` + `maneuver_timings_bot.yaml`, and a `--run` deploy
OVERWRITES the bot's copies with the repo's. So dashboard HSV/gain tuning done under the
`visual_lane_servoing` task is LOST on a project deploy — tune the repo files (or the bot's
files after deploy). HSV currently: yellow `h22-40` (kept narrow on purpose — `h120` reached
into the purple/pink surroundings and created false lane edges).

---

## 1. AprilTags — library now installed (pupil_apriltags) ✓

The bot now has **pupil_apriltags** (you installed it). The detector tries it first, so signs/
lights work. No action needed. (`dt_apriltags` fallback also in place if ever needed.)

## 1-old. (historical) AprilTags were disabled — no tag library on the bot

**Diagnosis (from the task log):**
`[AprilTagDetector] no AprilTag library available (tried pupil_apriltags, dt_apriltags;
last error: No module named 'dt_apriltags')`
→ the bot has **neither** library. The bot still lane-follows + brakes for duckies (YOLO),
but is **blind to AprilTag signs/lights** until one is installed.

**Code:** already handles either library (`apriltags.py` tries `pupil_apriltags` →
`dt_apriltags`). **No code change needed** — just install a library on the bot.

**FIX — on the bot (SSH):**
```bash
ssh franky@172.20.10.2
pip3 install dt-apriltags                 # Duckietown's lib (aarch64/3.6); first choice
python3 -c "import dt_apriltags; print('dt ok')"
#  if the build fails, try the other:
#    pip3 install pupil-apriltags
#  if BOTH fail to build (no wheel + no toolchain), install build deps then retry:
#    sudo apt-get install -y cmake build-essential
```
Then restart the task (no code redeploy needed — it re-imports on start):
```powershell
python launch.py --run --host 172.20.10.2 --task project
```
**Verify:** task log shows `[AprilTagDetector] using dt_apriltags`; hold a tag36h11 tag in
view → `GET http://172.20.10.2:5000/telemetry` shows it under `tags`.

---

## 2. Task hangs/idles when the camera view is open  ⇒ "drives a bit then stops"

**Diagnosis (proven):** the real `CameraDriver` reads `cv2.VideoCapture` with no lock, and
**two threads read it** — the agent loop AND the `/video` MJPEG stream. `cv2.VideoCapture`
is not thread-safe, so opening the camera view deadlocks the agent's `read()`. Evidence:
the agent loop timestamp `t` was frozen for 6.5 min while `/video` stayed live; `finally`
then zeroes the wheels → bot idle. (No traceback, not an FSM stop.)

**FIX — already implemented + deployed, NOT yet verified** (bot was taken mid-test):
- `agent.main(..., frame_observer=...)` — the agent hands each frame it reads to the server.
- `servers/project/real_server.py` — `/video` is built **only** from those frames
  (`_latest_frame`); the server never reads the camera itself. ⇒ single reader, no contention.

**TO DO when bot is free — VERIFY:** stream `/video` continuously and confirm the agent
loop keeps advancing:
```
# while hammering /video, /telemetry 't' must keep increasing (was frozen before)
```
**Extra robustness to add if it's still flaky:**
- Wrap the agent's per-loop body in `try/except` so one bad frame logs + continues instead
  of killing the thread.
- The Flask **dev server** is fragile under MJPEG load on a 4 GB Nano + hotspot — keep to
  **one** `/video` viewer, and/or drop the stream to ~10 fps. Don't open many tabs.

---

## 3. Camera locks on restart  ⇒ "no frames" / connection refused, needs a reboot

**Diagnosis:** killing a process that holds the CSI camera leaves `nvargus-daemon`'s capture
session stale → the next start gets `Camera opened but returned no frames` and flap-restarts.

**FIX — implemented in `duckiebot/camera_driver/camera_driver.py`** (outer retry: tear down
+ rebuild the GStreamer pipeline up to 6× with growing waits, so it rides out the
release race instead of needing a reboot). **BUT this file is NOT in the deploy bundle** —
the bot runs its own copy at `/home/franky/DuckieTown-Rewritten/`. To make it take effect:
- **Option A (preferred):** `git pull` the bot's repo so it gets the patched `camera_driver.py`.
- **Option B:** add `duckiebot/camera_driver` to `package_task()` in `launch.py` so it ships
  with the bundle. (Risk: overwrites the bot's hardware driver copy — only do if the bot's
  `duckiebot/` matches this repo.)

**Interim (no code):** if you see "no frames" / refused:
```bash
ssh franky@172.20.10.2 "sudo systemctl restart nvargus-daemon"   # frees the camera
#  or, surest:  ssh franky@172.20.10.2 "sudo reboot"
```
then **one** clean `--run`. Avoid rapid `--stop`/`--run` cycles — they re-trigger the lock.
NOTE: once #2 is verified (agent no longer hangs in `camera.read()`), `--stop` should release
the camera cleanly and this lock should largely stop happening on its own.

---

## Quick reference
- Deploy: `python launch.py --run --host <ip> --task project`  (bundles project + dep tasks + YOLO model)
- Stop:   `python launch.py --stop --host <ip>`
- Live:   `http://<ip>:5000/`  (debug HUD)  ·  `/video`  ·  `/telemetry`  ·  daemon `:8000/status` + `/logs`
- Order to bring up clean: reboot → `--run` once → check log line for apriltag backend → put on ground.
