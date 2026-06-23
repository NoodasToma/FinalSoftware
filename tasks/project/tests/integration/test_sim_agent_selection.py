"""Integration test — the SIM server (virtual_server) selects the right agent based
on the `sign_agent` flag in the sim maneuver_timings.yaml, and is byte-identical
(default false) to before. We capture the thread target without launching Godot."""

import threading

import pytest

vs = pytest.importorskip("servers.project.virtual_server")
import tasks.project.packages.agent as agent
import tasks.project.packages.agent_signs as agent_signs


class _FakeWheels:
    def set_wheels_speed(self, l, r): pass
    def poll_state(self): return (0.0, 0.0, 0.0)


@pytest.fixture
def captured(monkeypatch):
    """Replace threading.Thread (as used in _start_agent) with a capture so the agent
    target/kwargs are recorded but never actually started."""
    grabbed = {}

    class _CaptureThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
            grabbed['target'] = target
            grabbed['args'] = args
            grabbed['kwargs'] = kwargs or {}

        def start(self):
            grabbed['started'] = True

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(vs.threading, "Thread", _CaptureThread)
    monkeypatch.setattr(vs, "camera", object(), raising=False)
    monkeypatch.setattr(vs, "wheels", _FakeWheels(), raising=False)
    monkeypatch.setattr(vs, "leds", None, raising=False)
    # avoid the real TelemetryLogger touching the filesystem
    monkeypatch.setattr(vs, "TelemetryLogger", lambda *a, **k: (lambda snap: None))
    return grabbed


def _set_sign_agent(monkeypatch, tmp_path, value):
    cfg = tmp_path / "maneuver_timings.yaml"
    cfg.write_text(f"sign_agent: {str(value).lower()}\nbase_speed: 0.25\n")
    monkeypatch.setitem(vs.CONFIG_FILES, "maneuver_timings", str(cfg))


def test_default_runs_project_agent(captured, monkeypatch, tmp_path):
    _set_sign_agent(monkeypatch, tmp_path, False)
    vs._start_agent()
    assert captured['target'] is agent.main
    assert captured['started'] is True
    # the project agent gets the sim fidelity intrinsics
    assert 'apriltag_intrinsics' in captured['kwargs']


def test_sign_agent_true_runs_reference_agent(captured, monkeypatch, tmp_path):
    _set_sign_agent(monkeypatch, tmp_path, True)
    vs._start_agent()
    assert captured['target'] is agent_signs.main
    # the reference agent gets the sim timings + a drive_gate (sim manual drive),
    # and NOT the pupil-apriltags intrinsics (it uses its own cv2.aruco detector)
    assert 'timings' in captured['kwargs']
    assert 'drive_gate' in captured['kwargs']
    assert 'apriltag_intrinsics' not in captured['kwargs']


def test_both_paths_pass_observer_and_drive_gate(captured, monkeypatch, tmp_path):
    for val in (False, True):
        _set_sign_agent(monkeypatch, tmp_path, val)
        vs._start_agent()
        assert 'observer' in captured['kwargs']
        assert 'drive_gate' in captured['kwargs']
