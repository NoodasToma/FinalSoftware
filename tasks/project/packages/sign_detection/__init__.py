"""Ported from KvatiTown's working sign_detection package (branch
implementing-duck-detection). This is the reference's proven sign/intersection
behaviour: an AprilTag + red-line driven FSM that wraps the lane-servoing agent and
OVERRIDES its wheel commands while handling a sign/intersection, then hands control
back to lane-following. See sign_behavior.SignBehaviorFSM and agent_with_signs.

Why it's here: our own intersection logic kept letting lane-following override the
sign decision (the bot chose a turn but the white line pulled it the wrong way). The
reference solves that by EXECUTING the turn open-loop (fixed wheel speeds for a fixed
number of frames, lane ignored) once a red line is reached WITH a saved sign tag.
"""
