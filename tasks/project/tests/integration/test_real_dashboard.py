"""Integration tests — the REAL-BOT dashboard server (servers/project/real_server.py)
and the bot-safe manual-drive mechanism in the agent.

real_server imports Pi-only `smbus2`, so we inject a fake before importing it. Then
we exercise every dashboard route a button calls (manual, drive, config GET/POST,
restart) via Flask's test client, with camera/wheels faked and the agent lifecycle
no-op'd. Separately we test that the agent's `manual_cmd` hook drives the wheels
THROUGH the agent's single writer (never a second I2C-writer thread) — the safety
invariant that keeps the bot from the OSError(121) bus crashes.
"""

import sys
import types

import numpy as np
import pytest


# --- inject a fake smbus2 so the Pi-only driver chain imports on a dev machine ---
if 'smbus2' not in sys.modules:
    _fake = types.ModuleType('smbus2')
    _fake.SMBus = object
    sys.modules['smbus2'] = _fake

rs = pytest.importorskip("servers.project.real_server")


class _FakeWheels:
    def __init__(self):
        self.history = []
    def set_wheels_speed(self, l, r):
        self.history.append((float(l), float(r)))
    @property
    def last(self):
        return self.history[-1] if self.history else None


class _FakeCam:
    def read(self): return True, np.zeros((480, 640, 3), np.uint8)
    def start(self): pass
    def stop(self): pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(rs, "camera", _FakeCam(), raising=False)
    monkeypatch.setattr(rs, "wheels", _FakeWheels(), raising=False)
    # no-op the agent lifecycle: we test route plumbing, not real agent threads
    monkeypatch.setattr(rs, "_start_agent", lambda: [])
    monkeypatch.setattr(rs, "_restart_agent", lambda: None)
    # reset manual state
    with rs._manual_lock:
        rs._manual.update({'on': False, 'left': 0.0, 'right': 0.0, 'last_cmd': 0.0})
    return rs.app.test_client()


# --------------------------------------------------------------- page + routes
def test_dashboard_page_has_controls():
    for handler in ("toggleManual", "restartAgent", "save("):
        assert handler in rs._PAGE, f"dashboard page missing {handler}"
    assert 'id="dpad"' in rs._PAGE
    assert 'id="cfg"' in rs._PAGE


@pytest.mark.parametrize("url", ["/", "/telemetry", "/config"])
def test_get_routes_ok(client, url):
    assert client.get(url).status_code < 500


# --------------------------------------------------------------- manual drive
def test_manual_toggle_and_drive_store_command(client):
    assert client.post("/manual", json={"on": True}).status_code == 200
    assert rs._manual['on'] is True
    r = client.post("/drive", json={"left": 0.3, "right": -0.3})
    assert r.status_code == 200
    # /drive STORES the command; it must NOT write the wheels itself (the agent does)
    assert (rs._manual['left'], rs._manual['right']) == (0.3, -0.3)
    assert rs.wheels.last is None, "real_server.drive() must not write wheels directly"
    assert client.post("/manual", json={"on": False}).status_code == 200
    assert rs._manual['left'] == rs._manual['right'] == 0.0


def test_drive_rejected_outside_manual(client):
    client.post("/manual", json={"on": False})
    assert client.post("/drive", json={"left": 0.5, "right": 0.5}).status_code == 409


def test_manual_command_is_clamped(client):
    client.post("/manual", json={"on": True})
    client.post("/drive", json={"left": 5.0, "right": -5.0})    # way over range
    assert abs(rs._manual['left']) <= rs._MANUAL_MAX
    assert abs(rs._manual['right']) <= rs._MANUAL_MAX


# --------------------------------------------------------------- _manual_cmd hook
def test_manual_cmd_off_returns_none():
    with rs._manual_lock:
        rs._manual.update({'on': False})
    assert rs._manual_cmd() is None       # agent drives normally


def test_manual_cmd_returns_held_command(monkeypatch):
    import time
    with rs._manual_lock:
        rs._manual.update({'on': True, 'left': 0.25, 'right': -0.25, 'last_cmd': time.time()})
    assert rs._manual_cmd() == (0.25, -0.25)


def test_manual_cmd_decays_to_zero_when_stale():
    import time
    with rs._manual_lock:
        rs._manual.update({'on': True, 'left': 0.4, 'right': 0.4,
                           'last_cmd': time.time() - (rs._MANUAL_TIMEOUT_S + 1.0)})
    # stale (dropped client) -> stop, but stay in manual mode
    assert rs._manual_cmd() == (0.0, 0.0)


# --------------------------------------------------------------- config edit
def test_config_get_returns_sections(client):
    cfg = client.get("/config").get_json()
    # the bot config sections the dashboard edits
    assert "lane_servoing_bot" in cfg or "maneuver_timings_bot" in cfg


def test_config_post_writes_and_restarts(client, tmp_path, monkeypatch):
    # point one section at a temp file so the test doesn't mutate the repo config
    f = tmp_path / "lane_bot.yaml"
    f.write_text("p_gain: 0.4\nsteer_smooth: 0.5\n")
    monkeypatch.setitem(rs.CONFIG_FILES, "lane_servoing_bot", str(f))
    restarted = {"n": 0}
    monkeypatch.setattr(rs, "_restart_agent", lambda: restarted.__setitem__("n", restarted["n"] + 1))

    r = client.post("/config?restart=1", json={"lane_servoing_bot": {"p_gain": "0.5"}})
    assert r.status_code == 200
    assert restarted["n"] == 1
    import yaml
    saved = yaml.safe_load(f.read_text())
    assert saved["p_gain"] == 0.5
    assert saved["steer_smooth"] == 0.5      # untouched key preserved


def test_config_post_no_restart_when_requested(client, tmp_path, monkeypatch):
    f = tmp_path / "hsv.yaml"
    f.write_text("yellow_lower_h: 15\n")
    monkeypatch.setitem(rs.CONFIG_FILES, "lane_hsv_bot", str(f))
    restarted = {"n": 0}
    monkeypatch.setattr(rs, "_restart_agent", lambda: restarted.__setitem__("n", restarted["n"] + 1))
    r = client.post("/config?restart=0", json={"lane_hsv_bot": {"yellow_lower_h": "14"}})
    assert r.status_code == 200
    assert restarted["n"] == 0                # HSV-live path: no restart
