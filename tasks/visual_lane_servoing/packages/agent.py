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
) -> Tuple[list, list, list, list]:
    """Returns (yellow_xs, white_xs, yellow_slices, white_slices).

    yellow_xs/white_xs are the COMPACT mean-x lists (only slices where the line was
    seen) used for the lane-centre error. yellow_slices/white_slices are SLICE-ALIGNED
    (length _NUM_SLICES, None where a line wasn't seen in that slice) so the curve
    detector can pair the two lines per slice and use the perspective-cancelling
    lane-CENTRE shift — see cuvrve_behavior.detect_curve."""
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []
    yellow_slices = [None] * _NUM_SLICES
    white_slices  = [None] * _NUM_SLICES

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            mx = int(np.mean(idx)); yellow_xs.append(mx); yellow_slices[i] = mx

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            mx = int(np.mean(idx)); white_xs.append(mx); white_slices[i] = mx

    return yellow_xs, white_xs, yellow_slices, white_slices


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
        # --- "best of both" ports from the reference's perfect-lane branch -------
        # All DEFAULT OFF so the sim + hardware-free tests are byte-identical; the
        # real-bot config (lane_servoing_config_bot.yaml) opts in. See the comments
        # on each below.
        #
        # Wrong-side white rejection: if the white line is detected to the LEFT of
        # the yellow line, that white is the oncoming lane's edge / an intersection
        # artefact, not OUR lane's right edge — following its midpoint would steer
        # the bot toward oncoming traffic. When enabled we discard that white and
        # track the yellow only. 0/false keeps the old midpoint-of-both behaviour.
        self.reject_white_wrong_side = bool(cfg.get('reject_white_wrong_side', False))
        # Wheel-speed FLOOR (anti-stiction): real DC motors don't move below a PWM
        # threshold, so a wheel commanded below min_wheel_speed during a turn just
        # stalls. When >0 we lift BOTH wheels so the slower one clears the floor
        # (preserving their difference = the steering), then rescale if the peak
        # exceeds 1. 0.0 = no floor (a no-op; sim has no stiction).
        self.min_wheel_speed     = float(cfg.get('min_wheel_speed', 0.0))
        # Steering EMA: exponentially smooth the steering command across frames to
        # damp detection jitter into a steady turn (reference uses 0.6). 0 disables
        # (steering passes straight through, the old behaviour).
        self.steer_smooth        = float(cfg.get('steer_smooth', 0.0))

        self.frame_count        = 0
        self._recovery_frames   = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._filtered_steering = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self.last_debug_info    = self._empty_debug_info(480, 640)

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            # Wrong-side white: white LEFT of yellow is the oncoming-lane edge /
            # intersection noise, not our lane's right edge. Discard it and track
            # yellow only, so we don't steer the lane centre toward oncoming.
            if self.reject_white_wrong_side and w_mean <= y_mean:
                error = w / 2.0 - (y_mean + self._lane_half_width)
                return float(np.clip(error / (w / 2.0), -1.0, 1.0))
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

    def _apply_wheel_floor(self, left: float, right: float):
        """Lift both wheels so the slower one clears min_wheel_speed (preserving
        their difference = steering), then rescale if the peak exceeds 1. A no-op
        when min_wheel_speed <= 0 (the sim default). Ported from the reference's
        perfect-lane branch to beat real-motor stiction during turns."""
        if self.min_wheel_speed > 0.0:
            lo = min(left, right)
            if lo < self.min_wheel_speed:
                shift = self.min_wheel_speed - lo
                left += shift
                right += shift
            peak = max(left, right)
            if peak > 1.0:
                left /= peak
                right /= peak
        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

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

        # On a detected curve, give the OUTER wheel extra drive so the bot actually
        # tracks the bend instead of drifting wide and crossing the outer line.
        # steering>0 turns left (outer wheel = right); steering<0 turns right
        # (outer wheel = left). Both use curve_boost — the old code multiplied the
        # right wheel by a hard-coded 5 for left turns only, an asymmetric slam that
        # could fling the bot across the lane.
        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= self.curve_boost
            else:
                left  *= self.curve_boost

        return self._apply_wheel_floor(left, right)

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

        yellow_xs, white_xs, yellow_slices, white_slices = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible        = left_det and right_det and not recovery
        # Perspective-cancelling curve detection on the lane-CENTRE shift (see
        # cuvrve_behavior.detect_curve): a single line's far-near shift is ~100 px on
        # a straight here from pure perspective, which the old per-line test mistook
        # for a curve and swerved off straights; the centre barely moves on a straight.
        is_curve, curve_dir = detect_curve(yellow_slices, white_slices, self.curve_threshold)

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        # Bias the target toward the yellow line so the bot rides the lane middle
        # rather than drifting to the white edge (see center_offset above).
        raw_error            = float(np.clip(raw_error + self.center_offset, -1.0, 1.0))
        self._filtered_error = 0.5 * self._filtered_error + 0.5 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        # Optional steering EMA (steer_smooth>0): damp frame-to-frame jitter into a
        # steady turn. Disabled (0) => steering passes straight through, unchanged.
        if self.steer_smooth > 0.0:
            self._filtered_steering = (self.steer_smooth * steering
                                       + (1.0 - self.steer_smooth) * self._filtered_steering)
            steering = self._filtered_steering
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
