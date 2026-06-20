from enum import Enum, auto
from typing import Dict, Tuple


class State(Enum):
    DRIVE = auto()
    APPROACH = auto()
    STOPPED = auto()
    WAIT = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    STRAIGHT_THROUGH = auto()
    SOFT_STOP = auto()


TRANSITIONS: Dict[Tuple[State, str], State] = {
    (State.DRIVE,            'see_stop_or_yield'):  State.APPROACH,
    (State.DRIVE,            'see_light'):          State.APPROACH,
    (State.DRIVE,            'see_intersection'):   State.APPROACH,
    (State.DRIVE,            'obstacle'):           State.SOFT_STOP,
    (State.APPROACH,         'at_stop_line'):       State.STOPPED,
    (State.APPROACH,         'obstacle'):           State.SOFT_STOP,
    (State.STOPPED,          'choose_turn_left'):   State.TURN_LEFT,
    (State.STOPPED,          'choose_turn_right'):  State.TURN_RIGHT,
    (State.STOPPED,          'choose_straight'):    State.STRAIGHT_THROUGH,
    (State.STOPPED,          'wait'):               State.WAIT,
    (State.WAIT,             'cleared'):            State.STOPPED,
    (State.TURN_LEFT,        'turn_done'):          State.DRIVE,
    (State.TURN_RIGHT,       'turn_done'):          State.DRIVE,
    (State.STRAIGHT_THROUGH, 'turn_done'):          State.DRIVE,
    (State.SOFT_STOP,        'obstacle_cleared'):   State.DRIVE,
}


def next_state(s: State, event: str) -> State:
    return TRANSITIONS.get((s, event), s)
