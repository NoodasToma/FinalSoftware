"""Telemetry + sim-fidelity config for the `project` simulation.

Two things live here so every sim entry point (dashboard server, behaviour
suite) stays consistent:

1. ``SIM_APRILTAG_INTRINSICS`` / ``SIM_TAG_SIZE_M`` — the sim camera's intrinsics
   and AprilTag physical size. Passing these into ``agent.main`` makes the
   AprilTag detector compute a REAL ``est_distance_m``, so the bot takes the
   SAME ``est_distance < 0.25 m`` stop-line code path it uses on hardware
   (instead of the sim-only pixel proxy). These are sim-only; the real robot
   reads its own calibrated ``camera_config.yaml``.

2. ``TelemetryLogger`` — the ``observer`` you hand to ``agent.main``. Each agent
   loop it receives a snapshot of what the agent perceived + decided, enriches
   it with the Godot bot POSE (x, z, heading) via ``wheels.poll_state()``,
   appends it to a JSONL file (and an in-memory list), keeps the latest for the
   dashboard, and on ``close()`` writes a compact summary. This is what lets us
   SEE, frame by frame, where the bot is and what it is reacting to — the thing
   that was missing when past "verified" bugs slipped through.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


# --- Sim camera fidelity config ----------------------------------------------
# Godot BotCamera renders a 640x480 SubViewport (scenes/robot/duckie_bot.tscn).
# cx, cy = image centre. fx = fy (square pixels). The focal length depends on
# Godot's fov convention, so it was CALIBRATED EMPIRICALLY (place a known tag at a
# known distance, fx = side_px * distance / tag_size) and pasted back here. The value
# below is the calibrated result; re-derive it if the camera fov/viewport changes.
_SIM_TAG_SIZE_DEFAULT = 0.13                       # unified sim AprilTag plane (metres) — "normal"-sized signs
# fx calibrated empirically: est_distance matched the true camera->tag distance within
# ~3% across 0.4-1.1 m. (Initial fov guess of 313 was wrong; the Godot camera's
# effective focal length is ~252 px.)
_SIM_APRILTAG_DEFAULT = (252.0, 252.0, 320.0, 240.0)   # (fx, fy, cx, cy)


def _load_sim_camera_config():
    """Read the SIM camera intrinsics + tag size from config/camera_intrinsics_sim.yaml
    (the explicit sim counterpart of the real bot's camera_intrinsics.yaml — see that
    file's header for why the two platforms are calibrated differently). Falls back to
    the constants above if the file is missing/partial, so the sim never breaks on a
    bad edit. Returns ((fx, fy, cx, cy), tag_size_m)."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "config", "camera_intrinsics_sim.yaml")
    intr = _SIM_APRILTAG_DEFAULT
    size = _SIM_TAG_SIZE_DEFAULT
    try:
        import yaml
        with open(os.path.normpath(path)) as fh:
            cfg = yaml.safe_load(fh) or {}
        src = cfg.get("intrinsics", {})
        if all(k in src for k in ("fx", "fy", "cx", "cy")):
            intr = (float(src["fx"]), float(src["fy"]),
                    float(src["cx"]), float(src["cy"]))
        if cfg.get("tag_size_m") is not None:
            size = float(cfg["tag_size_m"])
    except Exception:
        pass   # keep the constants — sim fidelity tuning, not safety-critical
    return intr, size


SIM_APRILTAG_INTRINSICS, SIM_TAG_SIZE_M = _load_sim_camera_config()


def sim_fidelity_kwargs() -> dict:
    """The kwargs every sim path should pass to ``agent.main`` so the bot runs
    the hardware perception code path (real est_distance)."""
    return {
        "apriltag_intrinsics": SIM_APRILTAG_INTRINSICS,
        "apriltag_tag_size": SIM_TAG_SIZE_M,
    }


class TelemetryLogger:
    """Observer for ``agent.main`` that logs pose + detections + reaction.

    Usage:
        log = TelemetryLogger(run_dir, wheels=wheels, label="stop_sign")
        agent.main(cam, wheels, None, stop, observer=log, **sim_fidelity_kwargs())
        ...
        log.close()
        print(log.summary())
    """

    def __init__(self, run_dir: str, wheels=None, label: str = "run",
                 pose_hz: float = 20.0, keep_records: bool = True) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.label = label
        self._wheels = wheels
        self._pose_period = 1.0 / max(pose_hz, 1.0)
        self._last_pose_t = 0.0
        self._pose = (0.0, 0.0, 0.0)                  # (heading_deg, x, z)
        self.path = os.path.join(run_dir, f"{label}.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")
        self.records: Optional[list] = [] if keep_records else None
        self.latest: Optional[dict] = None
        self._t0: Optional[float] = None
        self._n = 0
        # Prime the pose: poll_state lags a round-trip, so without this the first
        # record would carry the previous scenario's pose (e.g. a bogus start_pose).
        if self._wheels is not None and hasattr(self._wheels, "poll_state"):
            for _ in range(3):
                try:
                    self._pose = self._wheels.poll_state()
                except Exception:
                    pass
                time.sleep(0.03)

    # -- pose enrichment ------------------------------------------------------
    def _poll_pose(self) -> Optional[dict]:
        if self._wheels is None or not hasattr(self._wheels, "poll_state"):
            return None
        now = time.time()
        if now - self._last_pose_t >= self._pose_period:
            try:
                self._pose = self._wheels.poll_state()
            except Exception:
                pass
            self._last_pose_t = now
        h, x, z = self._pose
        return {"heading_deg": round(float(h), 2), "x": round(float(x), 4), "z": round(float(z), 4)}

    # -- observer interface ---------------------------------------------------
    def __call__(self, snap: dict) -> None:
        t = snap.get("t", time.time())
        if self._t0 is None:
            self._t0 = t
        rec = dict(snap)
        rec["rel_t"] = round(t - self._t0, 3)
        rec["pose"] = self._poll_pose()
        self.latest = rec
        if self.records is not None:
            self.records.append(rec)
        try:
            self._fh.write(json.dumps(rec) + "\n")
            self._n += 1
            if self._n % 25 == 0:
                self._fh.flush()
        except Exception:
            pass

    # -- digest ---------------------------------------------------------------
    def summary(self) -> dict:
        recs = self.records or []
        if not recs:
            return {"n": 0, "label": self.label}

        # State timeline: one entry each time the state changes.
        timeline, prev = [], None
        for r in recs:
            s = r.get("state")
            if s != prev:
                timeline.append({"t": r.get("rel_t"), "state": s, "via": r.get("event")})
                prev = s

        tags_seen: dict = {}
        light_seen: set = set()
        obstacle_frames = 0
        min_stop_dist = None          # closest a stop/yield/t-light tag got while braking
        for r in recs:
            for tg in r.get("tags", []):
                tags_seen[tg["id"]] = tg["meaning"]
                d = tg.get("est_distance_m")
                if d is not None and r.get("state") in ("APPROACH", "STOPPED"):
                    min_stop_dist = d if min_stop_dist is None else min(min_stop_dist, d)
            lc = (r.get("light") or {}).get("color")
            if lc:
                light_seen.add(lc)
            if r.get("obstacle_stop"):
                obstacle_frames += 1

        poses = [r["pose"] for r in recs if r.get("pose")]
        return {
            "label": self.label,
            "n": len(recs),
            "duration_s": round(recs[-1].get("rel_t", 0.0) - recs[0].get("rel_t", 0.0), 2),
            "states": [t["state"] for t in timeline],
            "state_timeline": timeline,
            "tags_seen": {int(k): v for k, v in sorted(tags_seen.items())},
            "light_colors_seen": sorted(light_seen),
            "obstacle_stop_frames": obstacle_frames,
            "min_stop_distance_m": (round(min_stop_dist, 3) if min_stop_dist is not None else None),
            "start_pose": poses[0] if poses else None,
            "end_pose": poses[-1] if poses else None,
            "jsonl": self.path,
        }

    def close(self) -> dict:
        summ = self.summary()
        try:
            with open(os.path.join(self.run_dir, f"{self.label}.summary.json"),
                      "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)
        except Exception:
            pass
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        return summ
