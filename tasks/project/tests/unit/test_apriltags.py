"""Unit tests — the AprilTag detector wrapper (perception/apriltags.py).

Rubric: "Use April tag software". These run the REAL ``pupil_apriltags`` backend
against synthetic frames that carry a genuine ``tag36h11`` marker, so detection,
the TagObservation shape, and the calibrated-vs-uncalibrated distance paths are all
exercised without hardware."""

import math

import numpy as np

from tasks.project.packages.perception.apriltags import AprilTagDetector, TagObservation
from conftest import blank_frame, add_apriltag


def test_a_backend_is_available():
    # pupil_apriltags (dev/sim) or dt_apriltags (bot) must load, else signs are
    # invisible. The .venv311 test env has pupil_apriltags.
    d = AprilTagDetector()
    assert d._backend is not None


def test_blank_frame_has_no_tags():
    d = AprilTagDetector()
    assert d.detect(blank_frame()) == []


def test_empty_frame_is_safe():
    d = AprilTagDetector()
    assert d.detect(None) == []
    assert d.detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_detects_a_real_tag_by_id():
    d = AprilTagDetector()
    frame = add_apriltag(blank_frame(), tag_id=8, size_px=140)   # 4-way-intersect
    obs = d.detect(frame)
    ids = [o.id for o in obs]
    assert 8 in ids


def test_observation_shape_and_uncalibrated_distance_is_inf():
    d = AprilTagDetector()                       # no intrinsics -> distance unknown
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=120)
    obs = [o for o in d.detect(frame) if o.id == 1]
    assert obs, "tag 1 not detected"
    o = obs[0]
    assert isinstance(o, TagObservation)
    assert o.side_length_px > 0
    assert 0 <= o.center_xy[0] <= 640 and 0 <= o.center_xy[1] <= 480
    assert math.isinf(o.est_distance_m)          # no metric distance without calibration


def test_calibrated_intrinsics_give_finite_distance():
    # Passing intrinsics (the sim does this) makes est_distance_m metric, so the
    # bot can use the est_distance stop path. Same detector, different calibration.
    d = AprilTagDetector(tag_size_m=0.13, intrinsics=(252.0, 252.0, 320.0, 240.0))
    frame = add_apriltag(blank_frame(), tag_id=1, size_px=160)
    obs = [o for o in d.detect(frame) if o.id == 1]
    assert obs
    assert math.isfinite(obs[0].est_distance_m)
    assert obs[0].est_distance_m > 0


def test_bigger_tag_reads_as_more_pixels():
    d = AprilTagDetector()
    small = [o for o in d.detect(add_apriltag(blank_frame(), 1, size_px=90)) if o.id == 1]
    big = [o for o in d.detect(add_apriltag(blank_frame(), 1, size_px=180)) if o.id == 1]
    # exactly one tag-1 detection each => we're comparing the SAME tag at two sizes,
    # not accepting a misdetect or a partial failure.
    assert len(small) == 1 and len(big) == 1
    assert big[0].side_length_px > small[0].side_length_px
    # the apparent size should scale roughly with the rendered size (180/90 = 2x)
    assert big[0].side_length_px > 1.5 * small[0].side_length_px
