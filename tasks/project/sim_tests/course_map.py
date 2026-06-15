"""Single source of truth for the `project` simulation test course.

Defines every station the bot is tested at, and the geometry of the sign bays in
the scene. Imported by:
  * tasks/project/sim_tests/course_suite.py  -> assertions (per-sign + continuous)
  * servers/project/virtual_server.py        -> dashboard per-sign "jump" buttons
  * tasks/project/sim_tests/build_strip.py    -> emits the .tscn nodes for the bays

so the test definitions and the scene geometry can never drift apart.

Two groups of stations:

1. LOOP stations -- reuse the proven, drivable loop already in project.tscn
   (lane-follow, the 4-way stop+sign, the traffic light, the duckie obstacle, the
   parked vehicle). Poses match the values behaviour_suite.py already verifies.

2. SIGN BAYS -- a grid of teleport-addressable bays in the bare field SOUTH of the
   loop (z > 8) that exercise every remaining traffic-sign type. Each bay is a
   short approach lane + a red stop line + a trigger sign (stop, or yield) + the
   sign-under-test. The bot is teleported to the top of the bay, drives down,
   stops at the line, and the legal-turn set it computes is asserted.

   GEOMETRY NOTE (important): sim AprilTags decode out to only ~3 m, so the bays
   are spaced 4 m apart in BOTH x and z. That guarantees the bot can never decode
   a neighbouring bay's tag -- otherwise the agent's per-approach sign
   accumulation (agent.py `intersection_signs`) would be polluted by a foreign
   sign and the legal-turn assertion would be wrong.

Tag IDs -> meaning (apriltagsDB.yaml): 1 stop, 2 yield, 8 4-way, 11 T-int,
9 right-T, 10 left-T, 6 oneway-right, 7 oneway-left, 3 no-right, 4 no-left,
5 do-not-enter, 74 t-light-ahead, 400 vehicle, 95 duck-crossing, 12 pedestrian,
125 parking.
"""
from __future__ import annotations

ALL_TURNS = {"left", "right", "straight"}

# --- sign-bay grid geometry (field south of the loop) ------------------------
# Spacing > the ~3 m sim tag-decode range in BOTH axes so the bot can never decode
# a neighbouring bay's tag (4.5 m in x, 4 m in z; signs straddle the lane at ~0.3 m
# so worst-case neighbour sign distance still > 3 m).
GRID_X = [1.5, 6.0, 10.5]           # column (tile) centres, 4.5 m apart
GRID_LINE_Z = [9.0, 13.0, 17.0]    # stop-line z for each row, 4 m apart
LANE_OFFSET = 0.15                  # southbound right lane = tile centre + 0.15
SIGN_DZ = 0.10                      # sign sits this far NORTH of the line (toward the bot)
SIGN_X_OFFSET = 0.30               # signs straddle the lane: trigger +0.30, test -0.30
APPROACH_TILES = 4                  # straight tiles north of the line, per bay (~2.4 m road)
TELEPORT_DZ = 1.3                   # teleport this far north of the line (short, reliable approach)


def bay_geom(col: int, row: int) -> dict:
    """Geometry for the bay at grid (col, row)."""
    x = GRID_X[col]
    line_z = GRID_LINE_Z[row]
    lane_x = round(x + LANE_OFFSET, 3)
    return {
        "x": x,                                          # road-tile centre
        "lane_x": lane_x,                                # the lane the bot drives
        "line_z": line_z,
        "teleport": (lane_x, round(line_z + TELEPORT_DZ, 3), 0.0),
        "tile_zs": [round(line_z + 0.3 + 0.6 * i, 3) for i in range(APPROACH_TILES)],
        "sign_z": round(line_z + SIGN_DZ, 3),
        # red line centred on the LANE (matches the proven loop's stop lines)
        "line_x": lane_x,
        # signs straddle the lane so both stay in view on approach
        "trigger_x": round(lane_x + SIGN_X_OFFSET, 3),
        "test_x": round(lane_x - SIGN_X_OFFSET, 3),
    }


# name -> (grid cell, trigger tag, test tag, expected legal-turn set)
# trigger tag 1 = stop (no turn constraint), 2 = yield (no turn constraint).
# For the yield bay the trigger IS the sign under test, so test tag is None.
SIGN_BAYS = [
    # name            cell    trig  test  expected legal turns
    ("yield",        (0, 0),  2,   None, ALL_TURNS),
    ("t_intersect",  (1, 0),  1,   11,   {"left", "right"}),
    ("right_t",      (2, 0),  1,   9,    {"right", "straight"}),
    ("left_t",       (0, 1),  1,   10,   {"left", "straight"}),
    ("oneway_right", (1, 1),  1,   6,    {"right"}),
    ("oneway_left",  (2, 1),  1,   7,    {"left"}),
    ("no_right",     (0, 2),  1,   3,    {"left", "straight"}),
    ("no_left",      (1, 2),  1,   4,    {"right", "straight"}),
    ("do_not_enter", (2, 2),  1,   5,    set()),
]

# Roadside DECODE-ONLY signs placed on the west spawn straight (col1, x~0.9).
# The bot decodes them while lane-following past; no behaviour change expected.
# (x, z, tag, name)  -- placed on the right roadside (x>0.9), facing +Z.
DECODE_SIGNS = [
    (1.18, 4.5, 95,  "duck_crossing"),
    (1.18, 3.9, 12,  "pedestrian"),
    (1.18, 3.3, 125, "parking"),
]


def _bay_station(name, cell, trig, test, legal):
    g = bay_geom(*cell)
    tags = [trig] + ([test] if test is not None else [])
    label = name.replace("_", " ")
    if not legal:
        kind, expect = "do_not_enter", {"tags": tags}
    else:
        kind, expect = "legal", {"tags": tags, "legal": sorted(legal)}
    return {
        "name": name, "label": label, "teleport": g["teleport"],
        "seconds": 18, "kind": kind, "expect": expect,
    }


# ---------------------------------------------------------------------------- #
# The full station list the suite + dashboard iterate over.
STATIONS: list[dict] = [
    # --- LOOP stations (reuse proven poses from behaviour_suite) -------------
    {"name": "lane", "label": "lane following", "teleport": (0.9, 5.1, 0.0),
     "seconds": 8, "kind": "lane", "expect": {"max_err": 0.7, "min_move": 0.25}},
    {"name": "stop4way", "label": "stop + 4-way", "teleport": (5.85, 7.05, 0.0),
     "seconds": 18, "kind": "legal", "expect": {"tags": [1, 8], "legal": ["left", "right", "straight"]}},
    {"name": "light", "label": "traffic light", "teleport": (5.85, 4.45, 0.0),
     "seconds": 22, "kind": "light", "expect": {"tags": [74]}},
    {"name": "obstacle", "label": "duckie obstacle", "teleport": (4.5, 1.632, 270.0),
     "seconds": 9, "kind": "obstacle", "expect": {}},
    {"name": "vehicle", "label": "other robot (vehicle plate)", "teleport": (5.85, 4.85, 0.0),
     "seconds": 8, "kind": "vehicle", "expect": {"min_tag": 400}},
]

# --- DECODE-only roadside signs ---------------------------------------------
for _x, _z, _tag, _name in DECODE_SIGNS:
    # teleport SOUTH of the spawn-straight duckie (Duck_1_9 @ z=5.7) so the decode
    # drive-past isn't pre-empted by an obstacle soft-stop.
    STATIONS.append({
        "name": _name, "label": _name.replace("_", " "),
        "teleport": (0.9 + LANE_OFFSET, round(_z + 0.9, 3), 0.0),
        "seconds": 8, "kind": "decode", "expect": {"tag": _tag},
    })

# --- SIGN BAYS (every remaining turn-constraint sign) -----------------------
for _name, _cell, _trig, _test, _legal in SIGN_BAYS:
    STATIONS.append(_bay_station(_name, _cell, _trig, _test, _legal))


# Continuous drive-through: spawn and lane-follow the loop from the start.
COURSE_START = (0.9, 5.1, 0.0)


def station(name: str) -> dict | None:
    for s in STATIONS:
        if s["name"] == name:
            return s
    return None


if __name__ == "__main__":
    # Quick human-readable dump.
    for s in STATIONS:
        print(f"{s['name']:16s} {s['kind']:12s} tp={s['teleport']} expect={s['expect']}")
