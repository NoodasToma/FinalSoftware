from typing import Tuple
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Left motor weight matrix: high weights at bottom-left.
    Duck on left → high left motor signal → turns right (away).
    """
    h, w = shape
    i = np.arange(h).reshape(h, 1) / (h - 1)   # 0.0 top, 1.0 bottom
    j = np.arange(w).reshape(1, w) / (w - 1)   # 0.0 left, 1.0 right
    return 1 - (i + j) / 2


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Right motor weight matrix: high weights at bottom-right.
    Duck on right → high right motor signal → turns left (away).
    """
    h, w = shape
    i = np.arange(h).reshape(h, 1) / (h - 1)
    j = np.arange(w).reshape(1, w) / (w - 1)
    return 1 - (i + (1 - j)) / 2