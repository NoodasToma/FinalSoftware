import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain              = cfg.get('p_gain',              0.1)
        self.d_gain              = cfg.get('d_gain',              0.35)
        self.max_steer           = cfg.get('max_steer',           0.4)
        self.base_speed          = cfg.get('base_speed',          0.2)
        self.curve_speed         = cfg.get('curve_speed',         0.2)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.steering_threshold  = cfg.get('steering_threshold',  0.2)
        self.curve_boost         = cfg.get('curve_boost',         1.3)
        self.detection_threshold = cfg.get('detection_threshold', 500)
        # Lane-loss recovery: dead-stopping freezes forever (a stopped bot's
        # view never changes, so the lane can't reappear), but a hard pivot can
        # spin in a circle if the lane is truly gone. So recover in two bounded
        # phases (see _motor_commands): a brief GENTLE sweep toward the last-seen
        # lane side, then straight creep — never a tight in-place spin.
        self.recovery_speed      = cfg.get('recovery_speed',      0.15)
        self.recovery_turn       = cfg.get('recovery_turn',       0.08)
        self.recovery_max_frames = int(cfg.get('recovery_max_frames', 12))
        # Lateral target bias, normalized [-1, 1]. The lane center is the midpoint
        # of the yellow + white lines, but the dashed yellow / sparsely-detected
        # white make that estimate drift slightly toward the white edge when only
        # one line is seen. A POSITIVE center_offset shifts the target toward the
        # yellow (steer a touch left), so the bot rides the middle instead of
        # hugging the white edge. Tunable live via /update_config {"offset": ...}.
        self.center_offset       = cfg.get('center_offset',       0.0)

        self.frame_count        = 0
        self._recovery_frames   = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self.last_debug_info    = self._empty_debug_info(480, 640)

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0
        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)
        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)
        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            # Lane lost. Two bounded phases, so we neither freeze nor spin:
            #  1) for the first recovery_max_frames, GENTLY steer toward the side
            #     the lane was last seen (sign of _prev_error; >0 => lane was
            #     left) while creeping, so the camera sweeps back over it;
            #  2) after that, go STRAIGHT and keep creeping — translating to
            #     re-find the lane instead of circling in place.
            # The turn is deliberately gentle (recovery_turn ~0.08) so even the
            # sweep phase is a wide arc, never a tight pivot.
            self._recovery_frames += 1
            if self._recovery_frames <= self.recovery_max_frames:
                turn = self.recovery_turn if self._prev_error >= 0 else -self.recovery_turn
            else:
                turn = 0.0
            left  = self.recovery_speed - turn
            right = self.recovery_speed + turn
            return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

        self._recovery_frames = 0
        speed = self.curve_speed if is_curve else self.base_speed
        
        if not both_visible:
            speed *= 0.8

        left  = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= 5
            else:
                left  *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history)  / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)
        self.last_debug_info = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
        }

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0
        recovery  = total_pixels  < self.detection_threshold

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible        = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        # Bias the target toward the yellow line so the bot rides the lane middle
        # rather than drifting to the white edge (see center_offset above).
        raw_error            = float(np.clip(raw_error + self.center_offset, -1.0, 1.0))
        self._filtered_error = 0.5 * self._filtered_error + 0.5 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        left, right          = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right          = self._smooth(left, right, both_visible)

        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info.update({
            'yellow_xs': yellow_xs,
            'white_xs':  white_xs,
            'slice_ys':  [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve':  is_curve,
            'curve_dir': curve_dir,
        })

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
        }
