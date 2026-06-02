"""
Project task - Godot simulation server.

Mirrors servers/project/real_server.py, but swaps the hardware drivers for the
Godot TCP-backed drivers so the SAME tasks.project.packages.agent.main() runs
against the simulation instead of the physical Duckiebot.

Launched by:
    python launch.py --sim --task project

launch.py starts Godot on res://scenes/maps/project.tscn (lanes + stop signs +
AprilTags + traffic lights + duckies + an NPC bot), then runs this server. The
agent drives the simulated bot exactly as it would the real one; watch the
Godot window for behaviour and http://localhost:<port>/video for its camera POV.
"""

import sys
import os
import signal
import threading
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import numpy as np
import cv2
from flask import Flask, Response, jsonify

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

import tasks.project.packages.agent as agent


app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()


def _visualize(frame):
    """Frame generator callback. Godot frames are BGR (rgb=False below)."""
    if frame is not None:
        return frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for Godot camera...", (110, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
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

    ap = argparse.ArgumentParser(description='Project Server - Godot Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER - GODOT SIMULATION')
    print('=' * 60)

    print('\n[1/4] LED driver...')
    # Godot doesn't render the bot's LEDs, and the duckiebot.led_driver package
    # pulls in smbus2 (a Pi-only I2C lib not present off-bot). The agent guards
    # every LED call with `if leds:`, so None is the clean choice for the sim -
    # the LED policy no-ops here (it runs for real via real_server.py).
    leds = None
    print('  LEDs: disabled in sim (None)')

    print('\n[2/4] Initializing wheels driver (Godot)...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver (Godot)...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, leds, stop_event),
        daemon=True,
        name='AgentThread',
    ).start()
    print('  agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        shutdown_cleanup(wheels, camera, stop_event)
        if leds:
            try:
                leds.release()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nVideo stream: http://localhost:{web_port}/video')
    print('Watch the Godot window for the bot driving the course.')
    print('Press Ctrl+C to stop\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, stop_event)
        if leds:
            try:
                leds.release()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
