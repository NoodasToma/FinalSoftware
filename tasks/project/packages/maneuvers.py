import time
from typing import Optional


def ramp_speed(current: float, target: float, max_step: float) -> float:
    """Move `current` toward `target` by at most `max_step`. Clamp to
    the same sign as target when crossing zero."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0 else -max_step)


def _outer_ticks(wheels, side: str) -> Optional[int]:
    encoders = getattr(wheels, 'encoders', None)
    if encoders is None:
        return None
    return (encoders.right if side == 'right' else encoders.left).ticks


def _await_motion(wheels, stop_event, outer_side: str, target_ticks: int,
                  seconds: float, start_ticks: Optional[int] = None) -> None:
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

    # SAFETY TIMEOUT. On the real bot the wheel encoders are frequently not
    # actually counting (GPIO encoders not wired / failed to init silently). Then
    # `outer.ticks` never changes, the tick target is never reached, and this loop
    # would block FOREVER -> the whole agent freezes the instant it turns at an
    # intersection (the exact "drove around then froze" failure). So cap the wait:
    # if the encoders don't deliver the turn within a sane wall-clock window, fall
    # back to a TIME-based turn (rotate for `seconds`) and finish. A real,
    # working-encoder turn completes well under the cap; the sim is unaffected
    # (its modelled encoder advances normally).
    deadline = time.time() + max(seconds, 0.1)   # time-based fallback duration
    hard_cap = time.time() + max(seconds * 3.0, 6.0)
    moved_any = False
    while not stop_event.is_set():
        moved = abs(outer.ticks - start_ticks)
        if moved >= target_ticks:
            return
        if moved > 2:
            moved_any = True
        now = time.time()
        # Encoders clearly dead (no ticks at all after the fallback window): stop
        # waiting on them and complete the turn by time instead.
        if not moved_any and now >= deadline:
            t_end = now + max(seconds, 0.1)
            while time.time() < t_end and not stop_event.is_set():
                time.sleep(0.01)
            return
        if now >= hard_cap:          # absolute safety: never block past this
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
