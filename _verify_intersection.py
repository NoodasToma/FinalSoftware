"""Pure-logic verification of the junction-sign -> stop+turn change.
No Godot needed: exercises _derive_event / next_state / merge_turn_constraints
directly with synthetic signs."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from types import SimpleNamespace as NS
import numpy as np

from tasks.project.packages.agent import _derive_event, _triggers_approach, _INTERSECTION_KINDS
from tasks.project.packages.states import State, next_state
from tasks.project.packages.sign_registry import SignSemantic, lookup
from tasks.project.packages.perception.intersection import merge_turn_constraints

LANE = np.ones((480, 640), np.uint8) * 255   # bottom-third nonzero -> pixel fallback "at line"
FH = 480


def obs(dist=float('inf'), px=80, cx=320):
    return NS(est_distance_m=dist, side_length_px=px, center_xy=(cx, 100), id=9)


def sem(kind):
    return SignSemantic(kind=kind, tag_type='TrafficSign', vehicle_name=None)


def check(name, cond, detail=''):
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (' :: ' + detail if detail else ''))
    if not cond:
        check.failed += 1
check.failed = 0


rt = sem('right-T-intersect')

# 1) DRIVE sees a right-T sign -> see_intersection -> APPROACH (bot: est inf)
ev = _derive_event(State.DRIVE, [(obs(), rt)], [], FH, LANE, ignore_signs=False, at_red_line=False)
check("DRIVE + right-T sign -> 'see_intersection'", ev == 'see_intersection', repr(ev))
check("  ...and that transitions DRIVE -> APPROACH", next_state(State.DRIVE, ev) == State.APPROACH)

# 1b) post-turn cooldown (ignore_signs) suppresses re-trigger on the same sign
ev_c = _derive_event(State.DRIVE, [(obs(), rt)], [], FH, LANE, ignore_signs=True, at_red_line=False)
check("DRIVE + ignore_signs cooldown -> no re-trigger", ev_c is None, repr(ev_c))

# 2) APPROACH + painted RED LINE -> at_stop_line -> STOPPED (primary, sign-agnostic)
ev2 = _derive_event(State.APPROACH, [(obs(), rt)], [], FH, LANE, at_red_line=True)
check("APPROACH + red line -> 'at_stop_line'", ev2 == 'at_stop_line', repr(ev2))
check("  ...and that transitions APPROACH -> STOPPED", next_state(State.APPROACH, ev2) == State.STOPPED)

# 2b) APPROACH, no red line, sign within stop_distance (calibrated) -> at_stop_line
ev2b = _derive_event(State.APPROACH, [(obs(dist=0.2), rt)], [], FH, LANE, at_red_line=False, stop_distance_m=0.25)
check("APPROACH + sign close (0.2m<0.25m) -> 'at_stop_line'", ev2b == 'at_stop_line', repr(ev2b))

# 2c) APPROACH, uncalibrated (est inf), tag big + lane pixels -> pixel fallback fires
ev2c = _derive_event(State.APPROACH, [(obs(dist=float('inf'), px=80), rt)], [], FH, LANE, at_red_line=False)
check("APPROACH + uncalibrated big tag -> pixel-fallback 'at_stop_line'", ev2c == 'at_stop_line', repr(ev2c))

# 3) STOPPED: legal turns from the right-T sign are exactly {straight, right}
turns = merge_turn_constraints([rt])
check("right-T sign -> legal turns {straight,right}", turns == {'straight', 'right'}, str(turns))
for choice, st in [('right', State.TURN_RIGHT), ('straight', State.STRAIGHT_THROUGH)]:
    ev_t = {'left': 'choose_turn_left', 'right': 'choose_turn_right', 'straight': 'choose_straight'}[choice]
    check("  STOPPED + choose_%s -> %s" % (choice, st.name), next_state(State.STOPPED, ev_t) == st)

# 4) the real DB entry for tag 9 is a right-T sign and now triggers the flow
s9 = lookup(9)
check("DB tag 9 is right-T-intersect", s9 is not None and s9.kind == 'right-T-intersect', getattr(s9, 'kind', None))
check("  ...and _triggers_approach(tag 9)", _triggers_approach(s9.kind))

# 5) every junction kind triggers; full 4-way -> all three turns
for k in sorted(_INTERSECTION_KINDS):
    ev_k = _derive_event(State.DRIVE, [(obs(), sem(k))], [], FH, LANE, ignore_signs=False, at_red_line=False)
    check("DRIVE + %-18s -> 'see_intersection'" % k, ev_k == 'see_intersection', repr(ev_k))
check("4-way sign -> legal turns {left,right,straight}",
      merge_turn_constraints([sem('4-way-intersect')]) == {'left', 'right', 'straight'})

# 6) REGRESSIONS: existing triggers unchanged; non-junction signs still don't stop
check("stop sign still -> 'see_stop_or_yield'",
      _derive_event(State.DRIVE, [(obs(), sem('stop'))], [], FH, LANE, ignore_signs=False, at_red_line=False) == 'see_stop_or_yield')
check("t-light-ahead still -> 'see_light'",
      _derive_event(State.DRIVE, [(obs(), sem('t-light-ahead'))], [], FH, LANE, ignore_signs=False, at_red_line=False) == 'see_light')
check("oneway-right does NOT trigger a junction stop",
      _derive_event(State.DRIVE, [(obs(), sem('oneway-right'))], [], FH, LANE, ignore_signs=False, at_red_line=False) is None)
check("no sign in view -> no event",
      _derive_event(State.DRIVE, [], [], FH, LANE, ignore_signs=False, at_red_line=False) is None)

print("\n%s" % ("ALL PURE-LOGIC CHECKS PASSED" if check.failed == 0 else "%d CHECK(S) FAILED" % check.failed))
sys.exit(1 if check.failed else 0)
