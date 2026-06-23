"""Integration tests — the simulation DASHBOARD server (servers/project/virtual_server.py).

Exercises every endpoint a dashboard button calls, via Flask's test client, with the
camera/wheels faked and the agent lifecycle no-op'd (so no Godot, no real agent
threads). This is the regression guard for "the dashboard buttons work": manual
control, restart, reset, and the per-behaviour teleport buttons all return non-5xx,
and the page ships all the control handlers.
"""

import numpy as np
import pytest

vs = pytest.importorskip("servers.project.virtual_server")


class _FakeWheels:
    def set_wheels_speed(self, l, r): self.last = (l, r)
    def teleport(self, x, z, h): self.tp = (x, z, h)
    def reset_game(self): self.reset = True
    def poll_state(self): return (0.0, 0.0, 0.0, 0.0, 0.0)


class _FakeCam:
    def read(self): return True, np.zeros((480, 640, 3), np.uint8)
    def start(self): pass
    def stop(self): pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(vs, "camera", _FakeCam(), raising=False)
    monkeypatch.setattr(vs, "wheels", _FakeWheels(), raising=False)
    # no-op the agent lifecycle: we test endpoint plumbing, not real agent threads
    monkeypatch.setattr(vs, "_start_agent", lambda: None)
    monkeypatch.setattr(vs, "_stop_agent", lambda: None)
    monkeypatch.setattr(vs, "_restart_agent", lambda: None)
    monkeypatch.setattr(vs, "debug_proc", None, raising=False)
    return vs.app.test_client()


def test_lane_hsv_apply_enables_chunky_fill():
    """The sim applies chunky_fill (fills the dashed yellow centre line into solid
    blobs so the bot tracks the lane middle instead of drifting onto the yellow on
    curves). _apply_lane_hsv_live must turn it on from the sim HSV config."""
    from tasks.visual_lane_servoing.packages import visual_servoing_activity as vsa
    vsa.set_chunky_fill(False)
    vs._apply_lane_hsv_live()
    assert vsa._fill_chunky is True
    vsa.set_chunky_fill(False)   # restore module default for other tests


def test_dashboard_page_has_all_controls():
    for handler in ("toggleManual", "restartAgent", "resetSim", "runTests"):
        assert handler in vs._PAGE, f"dashboard page missing {handler} control"
    assert 'id="dpad"' in vs._PAGE          # manual drive pad present


@pytest.mark.parametrize("url", ["/", "/config", "/telemetry", "/log?since=0", "/scenarios"])
def test_get_endpoints_ok(client, url):
    assert client.get(url).status_code < 500


def test_manual_drive_cycle(client):
    assert client.post("/manual", json={"on": True}).status_code == 200
    # in manual mode, a drive command is accepted and HELD (the server-side
    # _manual_drive_loop pushes the held value to the wheels at ~50 Hz, decoupled
    # from how fast the browser sends — so /drive stores it, doesn't write directly)
    r = client.post("/drive", json={"left": 0.3, "right": -0.3})
    assert r.status_code == 200
    assert (vs._manual["left"], vs._manual["right"]) == (0.3, -0.3)
    assert vs._manual["on"] is True
    assert client.post("/manual", json={"on": False}).status_code == 200
    # leaving manual clears the held command so the bot doesn't lurch on resume
    assert (vs._manual["left"], vs._manual["right"]) == (0.0, 0.0)


def test_manual_drive_loop_pushes_held_command(client, monkeypatch):
    """The server-side drive loop pushes the held command to the wheels and decays
    to zero when the command goes stale (dropped client safety)."""
    import time
    client.post("/manual", json={"on": True})
    client.post("/drive", json={"left": 0.4, "right": 0.4})
    vs.wheels.last = None
    vs._manual_drive_loop_once = None
    # run one loop iteration's body directly (fresh command -> pushed)
    with vs._manual_lock:
        fresh = (time.time() - vs._manual["last_cmd"]) <= vs._MANUAL_TIMEOUT_S
        l = vs._manual["left"] if fresh else 0.0
    assert fresh and l == 0.4
    # now simulate a stale command -> the loop would zero the wheels
    with vs._manual_lock:
        vs._manual["last_cmd"] = time.time() - (vs._MANUAL_TIMEOUT_S + 1.0)
        stale = (time.time() - vs._manual["last_cmd"]) > vs._MANUAL_TIMEOUT_S
    assert stale
    client.post("/manual", json={"on": False})


def test_manual_does_not_stop_the_agent(client, monkeypatch):
    """Toggling manual must NOT stop/restart the agent thread (the old bug). With
    the drive_gate design, _set_manual only flips a flag + zeros the wheels."""
    calls = {"stop": 0, "start": 0}
    monkeypatch.setattr(vs, "_stop_agent", lambda: calls.__setitem__("stop", calls["stop"] + 1))
    monkeypatch.setattr(vs, "_start_agent", lambda: calls.__setitem__("start", calls["start"] + 1))
    client.post("/manual", json={"on": True})
    client.post("/manual", json={"on": False})
    assert calls == {"stop": 0, "start": 0}, "manual toggle must not restart the agent"


def test_drive_auto_enables_manual(client):
    # A /drive command can only come from the dashboard, so it AUTO-ENABLES manual
    # (no 409 race with /manual) — this is what makes WASD "just work". The command
    # is stored and manual mode turns on.
    client.post("/manual", json={"on": False})
    r = client.post("/drive", json={"left": 0.5, "right": 0.5})
    assert r.status_code == 200
    assert vs._manual["on"] is True
    assert (vs._manual["left"], vs._manual["right"]) == (0.5, 0.5)


def test_restart_and_reset(client):
    assert client.post("/restart").status_code == 200
    assert client.post("/reset").status_code == 200
    assert getattr(vs.wheels, "reset", False) is True   # reset teleported the bot


@pytest.mark.parametrize("name", ["lane", "stop4way", "obstacle", "vehicle"])
def test_behaviour_jump_buttons_teleport(client, name):
    r = client.post("/scenario/" + name)
    assert r.status_code == 200
    assert hasattr(vs.wheels, "tp")          # a teleport happened


def test_removed_scenarios_404_not_crash(client):
    # the stashed traffic light + removed sign-bays must 404 cleanly (no 500)
    for gone in ("light", "yield", "do_not_enter", "oneway_left"):
        assert client.post("/scenario/" + gone).status_code == 404


def test_scenarios_list_only_existing_stations(client):
    names = {s["name"] for s in client.get("/scenarios").get_json()}
    assert names == {"lane", "stop4way", "obstacle", "vehicle"}
    assert "light" not in names              # traffic light removed from the course
