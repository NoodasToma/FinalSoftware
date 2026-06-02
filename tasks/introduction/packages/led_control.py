import colorsys
from typing import List


def set_turning_leds(direction: str) -> dict:
    """Set LEDs to indicate turning direction."""

    yellow = list(colorsys.hsv_to_rgb(60/360, 1.0, 1.0))  # yellow
    white  = [1.0, 1.0, 1.0]
    red    = list(colorsys.hsv_to_rgb(0/360, 1.0, 1.0))   # red
    off    = [0.0, 0.0, 0.0]

    patterns = {
        'left':    {0: yellow, 2: off,    3: yellow, 4: off},   # diagonal for left turn
        'right':   {0: off,    2: yellow, 3: off,    4: yellow},# diagonal for right turn
        'forward': {0: white,  2: white,  3: white,  4: white}, # all white
        'stop':    {0: red,    2: red,    3: red,    4: red},   # all red
    }

    return patterns[direction]