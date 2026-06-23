"""Ported from KvatiTown's agent_with_signs.py (imports adapted to this repo).

LaneServoingAgentWithSigns extends the project's LaneServoingAgent: it computes the
normal lane-follow command, then runs it through the SignBehaviorFSM, which OVERRIDES
the wheels while handling a sign/intersection (stop, yield, or an open-loop turn) and
otherwise passes the lane command through unchanged.

    left, right, sign_state = agent.compute_commands(image_rgb, detections)
"""

from typing import List, Optional, Tuple

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.project.packages.sign_detection.sign_behavior import SignBehaviorFSM
from tasks.project.packages.sign_detection.sign_behavior_config import SignBehaviorConfig


class LaneServoingAgentWithSigns(LaneServoingAgent):
    """Lane servoing + AprilTag/red-line sign behaviour (the reference's design)."""

    def __init__(self, config_path=None, sign_config=None, **kwargs):
        super().__init__(config_path=config_path)

        if sign_config is None:
            cfg = SignBehaviorConfig(**kwargs)
        elif isinstance(sign_config, dict):
            cfg = SignBehaviorConfig(**sign_config)
        else:
            cfg = sign_config

        self._sign_fsm = SignBehaviorFSM(config=cfg)

    def reset_behavior(self):
        if getattr(self, "_sign_fsm", None) is not None:
            self._sign_fsm.reset()
        for attr, val in (("_prev_error", 0.0), ("_filtered_error", 0.0),
                          ("_filtered_steering", 0.0)):
            if hasattr(self, attr):
                setattr(self, attr, val)
        for attr in ("_left_history", "_right_history"):
            if hasattr(self, attr):
                getattr(self, attr).clear()

    def compute_commands(self, image, detections=None):
        # type: (np.ndarray, Optional[List]) -> Tuple[float, float, object]
        base_left, base_right = super().compute_commands(image)
        left, right, state = self._sign_fsm.step(
            image, base_left, base_right, detections or [],
        )
        return left, right, state

    @property
    def sign_state(self):
        return self._sign_fsm.state_name

    @property
    def sign_debug(self):
        return self._sign_fsm.debug

    def step(self, image, wheels_driver, detections=None):
        left, right, _ = self.compute_commands(image, detections)
        wheels_driver.set_wheels_speed(left, right)
        return left, right
