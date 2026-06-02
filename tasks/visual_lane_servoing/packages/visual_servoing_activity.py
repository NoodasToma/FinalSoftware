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

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 2)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    mask_mag = (Gmag > 50).astype(np.uint8)

    mask_yellow_color = cv2.inRange(imghsv, _yellow_lower, _yellow_upper)
    mask_white_color  = cv2.inRange(imghsv, _white_lower,  _white_upper)

    # ignore sky / upper portion of image
    mask_horizon = np.zeros((h, w), dtype=np.uint8)
    mask_horizon[int(h * 0.4):, :] = 1

    # yellow = yellow color + edge + horizon (no half restriction so detection survives drifts)
    mask_yellow = (mask_horizon * mask_mag
                   * (mask_yellow_color > 0)).astype(float)

    # white = white color + edge + horizon, exclude far-left strip
    mask_white_area = np.ones((h, w), dtype=np.uint8)
    mask_white_area[:, : w // 4] = 0
    mask_white = (mask_horizon * mask_white_area * mask_mag
                  * (mask_white_color > 0)).astype(float)

    return mask_yellow, mask_white


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]),  'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]),  'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]),  'white_upper_v':  int(_white_upper[2]),
    }