from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

SPEED = 1
TURN = 0.5


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    left  = 0.0
    right = 0.0

    if keys_pressed['up']:
        left  += SPEED
        right += SPEED

    if keys_pressed['down']:
        left  -= SPEED
        right -= SPEED

    if keys_pressed['left']:
        left  -= TURN
        right += TURN

    if keys_pressed['right']:
        left  += TURN
        right -= TURN

    return left, right