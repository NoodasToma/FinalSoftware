"""
Live simulation perception verifier.

Connects to the running project simulation's MJPEG feed (the *exact* BGR frames
the agent's perception sees) and runs them through the REAL detectors:

    * AprilTagDetector   - every traffic sign / traffic-light-ahead / vehicle tag
    * TrafficLightDetector - red / yellow / green blob detection

It accumulates everything the bot detects while it drives the course and prints
a coverage report: which AprilTag IDs (and their meanings) were seen, and which
traffic-light colours fired. This is how we confirm the simulation actually
*exercises* every behaviour the bot needs on the real hardware.

Run it WHILE the sim is up (`python launch.py --sim --task project`):

    .venv311/Scripts/python.exe tasks/project/sim_tests/verify_perception.py \
        --url http://localhost:5000/video --seconds 45 --save-dir _sim_frames

Requires pupil_apriltags (installed in .venv311); under an interpreter without
it, AprilTag detection degrades to "no tags" and the harness says so loudly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from collections import Counter

import cv2
import numpy as np

# Make `tasks.project.packages...` importable when run from the repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tasks.project.packages.perception.apriltags import AprilTagDetector
from tasks.project.packages.perception.traffic_light import TrafficLightDetector
from tasks.project.packages.sign_registry import lookup


def _mjpeg_frames(url: str, timeout: float):
    """Yield decoded BGR frames from an MJPEG (multipart/x-mixed-replace) URL."""
    stream = urllib.request.urlopen(url, timeout=timeout)
    buf = b""
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        buf += chunk
        # Pull out every complete JPEG (SOI 0xFFD8 ... EOI 0xFFD9) in the buffer.
        while True:
            soi = buf.find(b"\xff\xd8")
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if soi == -1 or eoi == -1:
                break
            jpg = buf[soi:eoi + 2]
            buf = buf[eoi + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:5000/video",
                    help="MJPEG endpoint of the running sim (default :5000/video)")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="How long to observe the drive")
    ap.add_argument("--save-dir", default=None,
                    help="If set, save one sample frame per second here (PNG)")
    ap.add_argument("--connect-timeout", type=float, default=20.0,
                    help="Seconds to keep retrying the initial connection")
    args = ap.parse_args()

    tags = AprilTagDetector()
    light = TrafficLightDetector()
    light.arm()  # force-on so we report colours regardless of agent arming state

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # The sim/Godot may still be coming up; retry the connection for a while.
    deadline = time.time() + args.connect_timeout
    frames = None
    first = None
    last_err = None
    while time.time() < deadline:
        try:
            gen = _mjpeg_frames(args.url, timeout=10.0)
            first = next(gen)  # force the connection + first decode
            frames = gen
            break
        except Exception as e:  # noqa: BLE001 - report and retry
            last_err = e
            time.sleep(1.0)
    if frames is None or first is None:
        print(f"[verify] ERROR: could not connect to {args.url} ({last_err})")
        return 2

    tag_hits: Counter = Counter()
    color_hits: Counter = Counter()
    n_frames = 0
    n_with_tag = 0
    start = time.time()
    next_save = start

    def _process(frame):
        nonlocal n_frames, n_with_tag
        n_frames += 1
        observed = tags.detect(frame) or []
        if observed:
            n_with_tag += 1
        for obs in observed:
            sem = lookup(obs.id)
            meaning = (sem.kind or sem.tag_type or "?") if sem else "UNMAPPED"
            tag_hits[(obs.id, meaning)] += 1
        color = light.detect(frame)
        if color:
            color_hits[color] += 1
        return observed, color

    _process(first)
    for frame in frames:
        observed, color = _process(frame)
        now = time.time()
        if args.save_dir and now >= next_save:
            tagstr = "-".join(str(o.id) for o in observed) or "none"
            cv2.imwrite(os.path.join(args.save_dir,
                        f"f{n_frames:04d}_t{int(now-start):02d}s_tags-{tagstr}_lt-{color or 'none'}.png"),
                        frame)
            next_save = now + 1.0
        if now - start >= args.seconds:
            break

    elapsed = time.time() - start
    print("\n" + "=" * 64)
    print("SIM PERCEPTION COVERAGE REPORT")
    print("=" * 64)
    print(f"frames observed : {n_frames} in {elapsed:.1f}s "
          f"({n_frames / elapsed:.1f} fps), {n_with_tag} had >=1 tag")
    if tags._det is None:  # noqa: SLF001 - intentional health check
        print("WARNING: pupil_apriltags NOT available - tag detection was DISABLED.")
    print("\nAprilTags detected (id -> meaning : frame count):")
    if tag_hits:
        for (tid, meaning), c in sorted(tag_hits.items()):
            print(f"  tag {tid:>3}  {meaning:<18} x{c}")
    else:
        print("  (none)")
    print("\nTraffic-light colours detected (colour : frame count):")
    if color_hits:
        for col, c in sorted(color_hits.items()):
            print(f"  {col:<8} x{c}")
    else:
        print("  (none)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
