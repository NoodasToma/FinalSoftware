from typing import Tuple
import os
import yaml
import numpy as np

GAINS_FILE = os.path.join(os.path.dirname(__file_), '..', '..', '..', 'config', 'modcon_config.yaml')
try:
    with open(_GAINS_FILE) as _f:
        _g = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _g = {}

K_P = _g.get('k_P', 0.0)
K_I = _g.get('k_I', 0.0)
K_D = _g.get('k_D', 0.0)
MAX_OMEGA = _g.get('max_omega', 8.0)
MIN_OMEGA = -MAX_OMEGA


def PIDController(
    v_0: float,
    theta_ref: float,
    theta_hat: float,
    prev_e: float,
    prev_int: float,
    delta_t: float,
) -> Tuple[float, float, float, float]:

    # 1. Compute tracking error
    e = theta_ref - theta_hat

    # 2. Integral term (finite sum approximation)
    e_int = prev_int + e * delta_t

    # 3. Derivative term (finite difference)
    e_der = (e - prev_e) / delta_t if delta_t > 0 else 0.0

    # 4. PID control law
    omega = K_P * e + K_I * e_int + K_D * e_der

    # 5. Saturation (limit angular velocity)
    omega = max(MIN_OMEGA, min(MAX_OMEGA, omega))

    # 6. Output
    v = v_0

    return v, omega, e, e_int