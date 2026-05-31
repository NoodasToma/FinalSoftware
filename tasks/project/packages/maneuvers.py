from __future__ import annotations

import time


def ramp_speed(current: float, target: float, max_step: float) -> float:
    """Move `current` toward `target` by at most `max_step`. Clamp to
    the same sign as target when crossing zero."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0 else -max_step)


def _outer_ticks(wheels, side: str) -> int | None:
    encoders = getattr(wheels, 'encoders', None)
    if encoders is None:
        return None
    return (encoders.right if side == 'right' else encoders.left).ticks


def _await_ticks(wheels, stop_event, outer_side: str, target_ticks: int,
                 start_ticks: int | None = None) -> None:
    """Block until the outer wheel encoder has advanced target_ticks ticks
    (in either direction), polling stop_event every 0.02s. If wheels.encoders
    is None, fall back to time.sleep(target_ticks * 0.05) with stop_event check."""
    encoders = getattr(wheels, 'encoders', None)

    if encoders is None:
        end_time = time.time() + target_ticks * 0.05
        while time.time() < end_time:
            if stop_event.is_set():
                return
            time.sleep(0.02)
        return

    outer = encoders.right if outer_side == 'right' else encoders.left
    if start_ticks is None:
        start_ticks = outer.ticks

    while not stop_event.is_set():
        if abs(outer.ticks - start_ticks) >= target_ticks:
            return
        time.sleep(0.02)


def turn_left(wheels, stop_event, timings) -> None:
    start = _outer_ticks(wheels, 'right')
    wheels.set_wheels_speed(timings['turn_inner_speed'], timings['turn_outer_speed'])
    _await_ticks(wheels, stop_event, 'right', timings['turn_ticks'], start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)


def turn_right(wheels, stop_event, timings) -> None:
    start = _outer_ticks(wheels, 'left')
    wheels.set_wheels_speed(timings['turn_outer_speed'], timings['turn_inner_speed'])
    _await_ticks(wheels, stop_event, 'left', timings['turn_ticks'], start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)


def straight_through(wheels, stop_event, timings) -> None:
    start = _outer_ticks(wheels, 'left')
    s = timings['straight_speed']
    wheels.set_wheels_speed(s, s)
    _await_ticks(wheels, stop_event, 'left', timings['straight_ticks'], start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)
