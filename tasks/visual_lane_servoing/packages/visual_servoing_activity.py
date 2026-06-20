from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

# Chunky-dash fill. The yellow/white masks AND the Sobel EDGE mask (mask_mag) with
# colour, which keeps thin dashes whole but ERASES the flat low-gradient INTERIOR
# of a chunky/square dash (leaving a 1-2 px outline) — so on venues whose centre
# dashes are fat squares the lane follower under-counts yellow and falls back to
# the white edge. When this flag is on, the edge mask is DILATED so the interior
# between a dash's two edges is filled before the colour AND. Default OFF: the sim
# (thin dashes) is byte-identical; real_server flips it on for the bot only.
_fill_chunky = False

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 2)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    mask_mag = (Gmag > 50).astype(np.uint8)

    # Optionally dilate the edge mask so a chunky dash's flat interior (where the
    # gradient is ~0) is filled in, instead of being erased by the colour AND.
    edge = cv2.dilate(mask_mag, np.ones((9, 9), np.uint8)) if _fill_chunky else mask_mag

    mask_yellow_color = cv2.inRange(imghsv, _yellow_lower, _yellow_upper)
    mask_white_color  = cv2.inRange(imghsv, _white_lower,  _white_upper)

    # ignore sky / upper portion of image
    mask_horizon = np.zeros((h, w), dtype=np.uint8)
    mask_horizon[int(h * 0.4):, :] = 1

    # yellow = yellow color + edge + horizon (no half restriction so detection survives drifts)
    mask_yellow = (mask_horizon * edge
                   * (mask_yellow_color > 0)).astype(float)

    # white = white color + edge + horizon, exclude far-left strip
    mask_white_area = np.ones((h, w), dtype=np.uint8)
    mask_white_area[:, : w // 4] = 0
    mask_white = (mask_horizon * mask_white_area * edge
                  * (mask_white_color > 0)).astype(float)

    return mask_yellow, mask_white


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)

def set_chunky_fill(on):
    """Enable/disable dilating the edge mask to keep chunky-dash interiors (see
    _fill_chunky). real_server turns this on for the bot; the sim leaves it off."""
    global _fill_chunky
    _fill_chunky = bool(on)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]),  'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]),  'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]),  'white_upper_v':  int(_white_upper[2]),
    }