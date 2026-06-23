"""Unit tests — sign recognition by AprilTag ID (rubric: "Recognizing and observing
all traffic signs", "Traffic signs have numbers for identification").

Verifies the AprilTag-ID -> meaning lookup that turns a decoded tag36h11 number
into a behaviour ``kind`` the FSM reacts to."""

import pytest

from tasks.project.packages.sign_registry import lookup, SignSemantic, _REGISTRY


# (tag_id, expected_kind) — one representative ID per behaviour class, taken from
# the shipped apriltagsDB.yaml (552 entries). See BOT_BEHAVIOR.md §2.
REPRESENTATIVE = [
    (1, "stop"),
    (2, "yield"),
    (39, "yield"),
    (3, "no-right-turn"),
    (4, "no-left-turn"),
    (5, "do-not-enter"),
    (69, "do-not-enter"),
    (6, "oneway-right"),
    (42, "oneway-right"),
    (7, "oneway-left"),
    (43, "oneway-left"),
    (8, "4-way-intersect"),
    (9, "right-T-intersect"),
    (10, "left-T-intersect"),
    (11, "T-intersection"),
    (74, "t-light-ahead"),
    (200, "t-light-ahead"),
]


@pytest.mark.parametrize("tag_id,kind", REPRESENTATIVE)
def test_sign_id_maps_to_expected_kind(tag_id, kind):
    sem = lookup(tag_id)
    assert sem is not None, f"tag {tag_id} missing from DB"
    assert sem.kind == kind
    assert sem.tag_type == "TrafficSign"


def test_db_loaded_and_substantial():
    # The whole point of the registry is breadth: hundreds of Duckietown tags.
    assert len(_REGISTRY) > 500


def test_db_entries_are_structurally_valid():
    """Not just 'many entries' — every entry must be well-formed on the invariants
    the agent actually relies on: every Vehicle plate carries a name (precedence
    needs it), and every turn directive resolves to a valid subset of
    {left, right, straight}. (Blank/placeholder tags with empty kind+type are
    legitimate and simply ignored by the FSM, so type is NOT required.)"""
    valid_turns = {"left", "right", "straight"}
    for tag_id, sem in _REGISTRY.items():
        assert isinstance(tag_id, int)
        assert sem.available_turns <= valid_turns, f"tag {tag_id}: bad turns {sem.available_turns}"
        if sem.tag_type == "Vehicle":
            assert sem.vehicle_name, f"vehicle tag {tag_id} has no name"


def test_every_behaviour_kind_has_at_least_one_tag():
    # The bot must be able to encounter each sign behaviour, i.e. the DB actually
    # contains every kind the FSM reacts to.
    kinds = {sem.kind for sem in _REGISTRY.values()}
    for required in ("stop", "yield", "4-way-intersect", "T-intersection",
                     "right-T-intersect", "left-T-intersect", "t-light-ahead",
                     "oneway-left", "oneway-right", "no-left-turn", "no-right-turn",
                     "do-not-enter"):
        assert required in kinds, f"no tag in DB for behaviour '{required}'"


def test_unknown_tag_returns_none():
    assert lookup(99999) is None
    assert lookup(-1) is None


def test_vehicle_plates_carry_a_robot_name():
    # Tags 400..439 are other Duckiebots ("megabot01".."megabot40"), used for
    # the give-way / who-goes-first decision — not a road sign.
    sem = lookup(400)
    assert sem is not None
    assert sem.tag_type == "Vehicle"
    assert sem.vehicle_name == "megabot01"
    assert sem.kind == ""          # vehicles have no road-sign behaviour kind
    assert lookup(439).vehicle_name == "megabot40"


def test_available_turns_for_each_intersection_kind():
    # available_turns is what the intersection logic intersects to pick a turn.
    assert lookup(8).available_turns == {"left", "right", "straight"}   # 4-way
    assert lookup(11).available_turns == {"left", "right"}              # T-stem
    assert lookup(9).available_turns == {"straight", "right"}           # right-T
    assert lookup(10).available_turns == {"straight", "left"}           # left-T
    assert lookup(6).available_turns == {"right"}                       # oneway-right
    assert lookup(7).available_turns == {"left"}                        # oneway-left
    assert lookup(5).available_turns == set()                           # do-not-enter


def test_semantic_is_hashable_frozen():
    # SignSemantic is used in sets/dicts (intersection_signs keyed by id, etc.).
    s = SignSemantic(kind="stop", tag_type="TrafficSign", vehicle_name=None)
    assert s in {s}
    with pytest.raises(Exception):
        s.kind = "yield"           # frozen dataclass
