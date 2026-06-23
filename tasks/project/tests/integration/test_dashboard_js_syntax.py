"""Regression guard: the embedded dashboard <script> in the sim AND bot servers must
be valid JavaScript. A single bad escape (e.g. '\\n' written as '\n' in the Python
triple-quoted page string, which Python turns into a REAL newline mid-JS-string)
silently breaks the ENTIRE script -> every button + WASD dies while the backend still
works. That exact bug shipped once; this test catches it.

Validated with `node --check` when Node is installed (most dev machines); skipped
otherwise. Also does a Node-free check for the specific newline-in-string footgun."""

import re
import shutil
import subprocess
import sys
import types

import pytest


def _extract_script(page: str) -> str:
    m = re.search(r"<script>(.*)</script>", page, re.S)
    assert m, "dashboard page has no <script> block"
    return m.group(1)


def _sim_page():
    import servers.project.virtual_server as vs
    return vs._PAGE


def _bot_page():
    # real_server imports Pi-only smbus2; stub it so we can import on a dev machine
    if "smbus2" not in sys.modules:
        fake = types.ModuleType("smbus2"); fake.SMBus = object
        sys.modules["smbus2"] = fake
    import servers.project.real_server as rs
    return rs._PAGE


PAGES = {"sim": _sim_page, "bot": _bot_page}


@pytest.mark.parametrize("name", list(PAGES))
def test_no_raw_newline_inside_js_string(name):
    """The footgun that broke it: a real newline character inside a single-quoted JS
    string literal. After the fix the page must carry the escape sequence backslash-n,
    never a literal newline that immediately follows `+'`."""
    js = _extract_script(PAGES[name]())
    # `+'` followed directly by a real newline = an unterminated string literal.
    assert "+'\n" not in js, f"{name} dashboard JS has a raw newline inside a string (breaks the whole script)"


@pytest.mark.parametrize("name", list(PAGES))
def test_dashboard_js_parses_with_node(name):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed; raw-newline check above still guards the known footgun")
    js = _extract_script(PAGES[name]())
    # encode UTF-8 explicitly (the page contains non-latin1 glyphs like ▶); the
    # default Windows cp1252 would fail to encode them.
    proc = subprocess.run([node, "--check", "-"], input=js.encode("utf-8"),
                          capture_output=True)
    assert proc.returncode == 0, \
        f"{name} dashboard JS has a syntax error:\n{proc.stderr.decode('utf-8', 'replace')}"
