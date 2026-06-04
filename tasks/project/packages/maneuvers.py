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


def _await_motion(wheels, stop_event, outer_side: str, target_ticks: int,
                  seconds: float, start_ticks: int | None = None) -> None:
    """Block until a maneuver completes, on either platform.

    ENCODER path (``wheels.encoders`` present — the real DaguWheelsDriver AND the
    Godot sim, which now MODELS a 135-tick/rev encoder): wait until the outer
    wheel encoder advances ``target_ticks`` ticks — a physically calibrated,
    geometry-faithful rotation amount, independent of loop/CPU speed. Because the
    sim models the same encoder, a turn tuned to ~90° in sim is ~90° on the bot
    (same baseline + inner/outer ratio), modulo real wheel slip.

    SECONDS fallback (``wheels.encoders is None`` — e.g. a bot whose GPIO
    encoders failed to init): rotate for ``seconds`` of wall-clock instead.

    ``target_ticks`` (encoder ticks) and ``seconds`` (duration) are NOT
    interchangeable — that conflation was the original bug: 90 encoder ticks is a
    physical arc, but reusing 90 as ``90 * 0.05s`` was ~4.5 s ≈ several full
    spins. They are separate knobs (``turn_ticks`` vs ``turn_seconds``).
    Polls ``stop_event`` so a shutdown interrupts the maneuver promptly."""
    encoders = getattr(wheels, 'encoders', None)

    if encoders is None:
        end_time = time.time() + seconds
        while time.time() < end_time:
            if stop_event.is_set():
                return
            time.sleep(0.01)
        return

    outer = encoders.right if outer_side == 'right' else encoders.left
    if start_ticks is None:
        start_ticks = outer.ticks

    while not stop_event.is_set():
        if abs(outer.ticks - start_ticks) >= target_ticks:
            return
        time.sleep(0.01)


def _turn_cfg(timings, direction: str):
    """Per-direction turn parameters with fallback to the shared keys.

    A car-like intersection turn is a gradual forward ARC, and left/right arcs
    have different radii (right: tight-ish onto the near lane; left: wide across
    the junction onto the far lane). Radius is set by the inner/outer speed
    ratio: R = (baseline/2) * (outer+inner) / (outer-inner); ticks set how much
    of the arc is driven. All four knobs are per-direction in
    maneuver_timings.yaml (turn_left_*/turn_right_*), falling back to the legacy
    shared turn_* keys so old configs keep working."""
    return (
        timings.get(f'turn_{direction}_inner_speed', timings['turn_inner_speed']),
        timings.get(f'turn_{direction}_outer_speed', timings['turn_outer_speed']),
        timings.get(f'turn_{direction}_ticks',       timings['turn_ticks']),
        timings.get(f'turn_{direction}_seconds',     timings.get('turn_seconds', 0.4)),
    )


def turn_left(wheels, stop_event, timings) -> None:
    inner, outer, ticks, seconds = _turn_cfg(timings, 'left')
    start = _outer_ticks(wheels, 'right')
    wheels.set_wheels_speed(inner, outer)
    _await_motion(wheels, stop_event, 'right', ticks, seconds, start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)


def turn_right(wheels, stop_event, timings) -> None:
    inner, outer, ticks, seconds = _turn_cfg(timings, 'right')
    start = _outer_ticks(wheels, 'left')
    wheels.set_wheels_speed(outer, inner)
    _await_motion(wheels, stop_event, 'left', ticks, seconds, start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)


def straight_through(wheels, stop_event, timings) -> None:
    start = _outer_ticks(wheels, 'left')
    s = timings['straight_speed']
    wheels.set_wheels_speed(s, s)
    _await_motion(wheels, stop_event, 'left', timings['straight_ticks'],
                  timings.get('straight_seconds', 2.2), start_ticks=start)
    wheels.set_wheels_speed(0.0, 0.0)
