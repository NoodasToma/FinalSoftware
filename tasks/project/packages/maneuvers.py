from __future__ import annotations


def ramp_speed(current: float, target: float, max_step: float) -> float:
    """Move `current` toward `target` by at most `max_step`. Clamp to
    the same sign as target when crossing zero."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0 else -max_step)


def turn_left(wheels, stop_event, timings):
    raise NotImplementedError


def turn_right(wheels, stop_event, timings):
    raise NotImplementedError


def straight_through(wheels, stop_event, timings):
    raise NotImplementedError
