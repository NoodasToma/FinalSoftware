import sys
import os
import signal
import threading
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify
import numpy as np
import cv2
import yaml

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

import tasks.project.packages.agent as agent

# ---------------------------------------------------------------- bot configs
# The base YAMLs hold the SIM-verified values (the behaviour suite runs against
# them). The real robot is NOT the sim — in particular DaguWheelsDriver uses
# pwm_min=60 (stiction floor), which compresses wheel-speed ratios and would
# make the sim's turn arcs far wider on hardware — so the hardware starting
# values live in small *_bot.yaml overlays that only this server applies.
_PKG_CONFIG = os.path.join(project_root, 'tasks', 'project', 'packages', 'config')


def _load_bot_timings():
    """maneuver_timings.yaml merged with maneuver_timings_bot.yaml (if present)."""
    with open(os.path.join(_PKG_CONFIG, 'maneuver_timings.yaml')) as fh:
        timings = yaml.safe_load(fh) or {}
    overlay_path = os.path.join(_PKG_CONFIG, 'maneuver_timings_bot.yaml')
    overlay = {}
    try:
        with open(overlay_path) as fh:
            overlay = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        pass
    timings.update(overlay)
    return timings, sorted(overlay)


def _bot_lane_config_path():
    """config/lane_servoing_config_bot.yaml if it exists, else None (default)."""
    path = os.path.join(project_root, 'config', 'lane_servoing_config_bot.yaml')
    return path if os.path.isfile(path) else None

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()


def _visualize(frame):
    if frame is not None:
        return frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=False)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds, stop_event

    ap = argparse.ArgumentParser(description='Project Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting agent...')
    timings, overlay_keys = _load_bot_timings()
    lane_cfg = _bot_lane_config_path()
    print(f"  bot timing overrides: {overlay_keys or 'none'}")
    print(f"  bot lane config: {lane_cfg or 'default (sim-tuned!)'}")
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event),
        kwargs=dict(timings_override=timings, lane_config_path=lane_cfg),
        daemon=True,
        name='AgentThread',
    ).start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nVideo stream: http://localhost:{web_port}/video')
    print('Press Ctrl+C to stop\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
