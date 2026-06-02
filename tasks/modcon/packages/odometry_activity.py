from typing import Tuple
import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    # Calculate the change in wheel rotation (delta_phi) and the change in ticks (delta_ticks)
    delta_ticks = ticks - prev_ticks

    # Each tick corresponds to a certain angle of wheel rotation, which can be calculated using the resolution of the encoder.
    alpha = 2 * np.pi / resolution

    # Calculate the change in wheel rotation (delta_phi) based on the change in ticks and the angle per tick (alpha)
    delta_phi = alpha * delta_ticks

    return delta_phi, delta_ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    # distance travelled by wheels
    d_left = R * delta_phi_left
    d_right = R * delta_phi_right

    # robot frame distance
    d_A = (d_right + d_left) / 2

    # change in orientation
    delta_theta = (d_right - d_left) / (2 * baseline)

    # world frame motion
    theta = theta_prev + delta_theta
    delta_x = d_A * np.cos(theta)
    delta_y = d_A * np.sin(theta)

    # updated pose
    x = x_prev + delta_x
    y = y_prev + delta_y
    return x, y, theta
