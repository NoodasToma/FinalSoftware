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


_SLEEP = 0.01


def _await_motion(wheels, stop_event, outer_side: str, target_ticks: int,
                  seconds: float, start_ticks: Optional[int] = None,
                  tick=None, resume=None, set_wheels=None,
                  hold_max_s: float = 0.0, hold_state=None) -> None:
    """Block until a maneuver completes, on either platform.

    ENCODER path (``wheels.encoders`` present — the real DaguWheelsDriver AND the
    Godot sim, which now MODELS a 135-tick/rev encoder): wait until the outer
    wheel encoder advances ``target_ticks`` ticks — a physically calibrated,
    geometry-faithful rotation amount, independent of loop/CPU speed.

    SECONDS fallback (``wheels.encoders is None`` OR encoders not counting — the
    real bot's GPIO encoders are frequently dead): rotate for ``seconds`` of
    wall-clock instead, with a hard cap so a stuck turn can never freeze the agent.

    PERCEPTION-ALIVE TURNS. A turn used to BLOCK this loop with no camera reads, so
    the bot drove the intersection blind (no object detection). If ``tick`` is
    given it is called every iteration (in the SAME thread, so the single-I2C-writer
    rule holds) to keep the camera + object detector live; when it returns truthy
    (an obstacle ahead) the maneuver HOLDS — wheels to 0 via ``set_wheels``. Time is
    measured by the WALL CLOCK with held seconds subtracted out, so (a) the
    encoder-dead fallback/hard-cap fire at faithful real durations even though
    ``tick`` does a blocking camera.read() each iteration (crediting a fixed _SLEEP
    per loop instead under-counted real time ~4-5x), and (b) an obstacle hold does
    not burn the turn's time budget, so the arc resumes (re-applying ``resume``
    speeds) and completes where it left off. ``hold_max_s`` caps TOTAL hold time;
    pass a shared ``hold_state`` dict across a multi-segment maneuver (arc +
    straight-exit) so the cap is per-MANEUVER, not per-call. ``tick=None``
    reproduces the original blocking behaviour exactly (sim/legacy)."""
    if hold_state is None:
        hold_state = {'held_total': 0.0}
    # `hold_state['held_total']` is the CAP budget, shared across a multi-segment
    # maneuver. `_self_held` is THIS call's held wall-time, used to subtract holds
    # from this call's own elapsed (t0 is per-call, so _unheld must not subtract a
    # prior segment's holds carried in held_total).
    _self_held = [0.0]

    def _do_hold():
        """Run the perception tick; if it reports an obstacle (and we're under the
        shared hold cap) hold wheels at 0 and return True. Returns False otherwise,
        re-applying ``resume`` speeds (idempotent) so the arc continues."""
        if tick is None:
            return False
        try:
            blocked = bool(tick())
        except Exception:
            blocked = False                     # a bad frame must never freeze/kill the turn
        if blocked and (hold_max_s <= 0 or hold_state['held_total'] < hold_max_s):
            if set_wheels is not None:
                set_wheels(0.0, 0.0)
            return True
        if resume is not None and set_wheels is not None:
            set_wheels(resume[0], resume[1])    # re-apply speeds after a hold (idempotent)
        return False

    t0 = time.time()

    def _unheld():
        return (time.time() - t0) - _self_held[0]

    def _step():
        """One loop iteration: tick/hold, pace, and charge held WALL time to both
        this call's unheld accounting and the shared cap (so blocking camera.read()
        time inside tick() is accounted correctly). Returns True if it was a hold."""
        it = time.time()
        held = _do_hold()
        time.sleep(_SLEEP)
        if held:
            dt = time.time() - it
            _self_held[0] += dt
            hold_state['held_total'] += dt
        return held

    encoders = getattr(wheels, 'encoders', None)

    if encoders is None:
        while _unheld() < seconds and not stop_event.is_set():
            _step()
        return

    outer = encoders.right if outer_side == 'right' else encoders.left
    if start_ticks is None:
        start_ticks = outer.ticks

    # Time budgets use UNHELD wall-clock so (a) an obstacle hold can't exhaust the
    # encoder-dead fallback / hard cap and abandon the turn mid-arc, and (b) the
    # durations are faithful real seconds regardless of per-iteration tick cost.
    fallback_after = max(seconds, 0.1)
    hard_after = max(seconds * 3.0, 6.0)
    moved_any = False
    while not stop_event.is_set():
        moved = abs(outer.ticks - start_ticks)
        if moved >= target_ticks:
            return
        if moved > 2:
            moved_any = True
        # Encoders clearly dead (no ticks after the unheld fallback window): finish
        # by (unheld) time instead of waiting on ticks that never come.
        if not moved_any and _unheld() >= fallback_after:
            t_end = _unheld() + max(seconds, 0.1)
            while _unheld() < t_end and not stop_event.is_set():
                _step()
            return
        if _unheld() >= hard_after:       # absolute safety: never block past this (unheld) cap
            return
        _step()


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


def _straight_exit(wheels, stop_event, timings, direction: str,
                   tick=None, set_wheels=None, hold_max_s: float = 0.0,
                   hold_state=None) -> None:
    """Drive STRAIGHT out of the intersection box at the end of a turn, before the
    lane PD takes back over.

    A turn arc lands the bot still inside/at the edge of the junction box, where
    there are no clean outgoing-lane markings — only stray edges from the other
    arms. Handing straight back to lane-following there makes the PD chase a wrong
    edge and unwind the turn (the heading spirals back toward the entry heading and
    the bot cuts diagonally off the lane). A short fixed straight segment first
    carries the bot onto the outgoing lane's real markings, so the PD then has the
    right lane to lock onto. This mirrors the APPROACH "creep dead straight" stretch
    (lane paint ends at the box).

    Distance is a CALIBRATION knob (clear the box, ~half a tile), tuned on the bot
    like the turn arcs themselves — see BOT_BEHAVIOR.md §1b. It can be set
    per-direction (`turn_left_exit_ticks` / `turn_right_exit_ticks`) for arcs whose
    widths differ, falling back to the shared `turn_exit_ticks`. Tick-based =>
    same code path on sim and hardware; set the value to 0 to disable."""
    ticks = int(timings.get('turn_%s_exit_ticks' % direction,
                            timings.get('turn_exit_ticks', 0)))
    if ticks <= 0:
        return
    _sw = set_wheels or wheels.set_wheels_speed
    s = timings['straight_speed']
    start = _outer_ticks(wheels, 'left')
    _sw(s, s)
    _await_motion(wheels, stop_event, 'left', ticks,
                  timings.get('turn_exit_seconds', 0.8), start_ticks=start,
                  tick=tick, resume=(s, s), set_wheels=_sw, hold_max_s=hold_max_s,
                  hold_state=hold_state)


def turn_left(wheels, stop_event, timings, tick=None, set_wheels=None,
              hold_max_s: float = 0.0) -> None:
    _sw = set_wheels or wheels.set_wheels_speed
    # One shared hold budget for the whole turn (arc + straight-exit), so an
    # obstacle can't be held for hold_max_s on EACH segment (2x the intended cap).
    hold_state = {'held_total': 0.0}
    inner, outer, ticks, seconds = _turn_cfg(timings, 'left')
    start = _outer_ticks(wheels, 'right')
    _sw(inner, outer)
    _await_motion(wheels, stop_event, 'right', ticks, seconds, start_ticks=start,
                  tick=tick, resume=(inner, outer), set_wheels=_sw,
                  hold_max_s=hold_max_s, hold_state=hold_state)
    _straight_exit(wheels, stop_event, timings, 'left',
                   tick=tick, set_wheels=_sw, hold_max_s=hold_max_s, hold_state=hold_state)
    _sw(0.0, 0.0)


def turn_right(wheels, stop_event, timings, tick=None, set_wheels=None,
               hold_max_s: float = 0.0) -> None:
    _sw = set_wheels or wheels.set_wheels_speed
    hold_state = {'held_total': 0.0}        # shared arc+exit hold budget (see turn_left)
    inner, outer, ticks, seconds = _turn_cfg(timings, 'right')
    start = _outer_ticks(wheels, 'left')
    _sw(outer, inner)
    _await_motion(wheels, stop_event, 'left', ticks, seconds, start_ticks=start,
                  tick=tick, resume=(outer, inner), set_wheels=_sw,
                  hold_max_s=hold_max_s, hold_state=hold_state)
    _straight_exit(wheels, stop_event, timings, 'right',
                   tick=tick, set_wheels=_sw, hold_max_s=hold_max_s, hold_state=hold_state)
    _sw(0.0, 0.0)


def straight_through(wheels, stop_event, timings, tick=None, set_wheels=None,
                     hold_max_s: float = 0.0) -> None:
    _sw = set_wheels or wheels.set_wheels_speed
    start = _outer_ticks(wheels, 'left')
    s = timings['straight_speed']
    _sw(s, s)
    _await_motion(wheels, stop_event, 'left', timings['straight_ticks'],
                  timings.get('straight_seconds', 2.2), start_ticks=start,
                  tick=tick, resume=(s, s), set_wheels=_sw, hold_max_s=hold_max_s)
    _sw(0.0, 0.0)
