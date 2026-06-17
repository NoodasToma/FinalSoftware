#!/bin/bash
#
# DuckieBot AprilTag library — system-wide install (Jetson Nano, Python 3.6)
#
# WHY THIS EXISTS / THE BUG IT FIXES
# ----------------------------------
# AprilTag detection (signs, traffic lights, other-bot tag plates) needs a Python
# tag library — `pupil_apriltags`. The catch on this bot: the task is launched by
# the dashboard daemon (duckiebot/start_bot.py) **as root**. A normal
# `pip3 install --user pupil-apriltags` lands in /home/<user>/.local, which root's
# Python does NOT search — so inside the running task `import pupil_apriltags`
# fails, AprilTagDetector silently falls back to "no library", and EVERY tag goes
# undetected (tags=0, no sign boxes on the debug screen) even though the same
# import works fine from your SSH shell. The fix is to install it into the SYSTEM
# site-packages (/usr/local/lib/python3.6/dist-packages), which is on sys.path for
# every Python process regardless of user or $HOME.
#
# This mirrors duckiebot/button_driver/install.sh, which does the same
# user-build -> system-install dance for dataclasses/pyserial.
#
# Usage (on the bot):  bash duckiebot/apriltags/install.sh
# Idempotent: safe to re-run; it verifies and exits early if root can already import.

set -e

echo "=========================================="
echo "DuckieBot AprilTag library install"
echo "=========================================="

if [ ! -f "/proc/device-tree/model" ]; then
    echo "Warning: not running on a Jetson — continuing anyway."
fi

PKG="pupil_apriltags"

# The import check MUST run in ROOT's own environment (HOME=/root), because that
# is the context the dashboard daemon launches the task in. `sudo -H` forces that.
# (Plain `sudo`, which keeps HOME=/home/<user>, can even segfault if a stale
# per-user copy is on the path — see the uninstall step below.)
ROOT_PY="sudo -H python3"

echo ""
echo "[1/5] Is ${PKG} already importable BY ROOT (the context the task runs in)?"
if $ROOT_PY -c "import ${PKG}" 2>/dev/null; then
    echo "  ✓ root can already import ${PKG} — nothing to do."
    $ROOT_PY -c "import ${PKG}; print('    ->', ${PKG}.__file__)"
    exit 0
fi
echo "  not yet — installing system-wide."

echo ""
echo "[2/5] Ensure a modern pip for the source build (JetPack ships pip 9, which"
echo "      can't PEP517-build this package). Upgrading the per-user pip only."
python3 -m pip install --user "pip<21.4" >/dev/null 2>&1 || true

echo ""
echo "[3/5] BUILD the aarch64/cp36 wheel (compiles the C extension; needs cmake +"
echo "      gcc, both present on JetPack 4.6). 'pip wheel' builds WITHOUT installing,"
echo "      so it never creates a per-user copy that would later clash system-wide."
rm -rf /tmp/apriltag_whl && mkdir -p /tmp/apriltag_whl
python3 -m pip wheel --no-deps -w /tmp/apriltag_whl "${PKG//_/-}"
WHL="$(find /tmp/apriltag_whl -name 'pupil_apriltags*.whl' 2>/dev/null | head -1)"
if [ -z "$WHL" ]; then
    echo "  ✗ wheel build failed. Install the toolchain and retry:"
    echo "      sudo apt-get install -y cmake build-essential"
    exit 1
fi
echo "  built: $WHL"

echo ""
echo "[4/5] Install that wheel into SYSTEM dist-packages so ROOT (the daemon) sees"
echo "      it.  --no-deps: numpy already present system-wide; don't let old root-pip"
echo "      rebuild it.  -H: root's HOME, so pip targets the system location."
sudo -H pip3 install --no-deps --ignore-installed "$WHL"

echo ""
echo "[5/5] Remove any per-user copy (~/.local). Having BOTH a --user and a system"
echo "      copy makes Python load two builds of the bundled libapriltag.so, which"
echo "      SEGFAULTS on import. Keep exactly one (the system one)."
python3 -m pip uninstall -y "${PKG//_/-}" >/dev/null 2>&1 || true

echo ""
echo "Verifying root can import it (this is the check that actually matters):"
if $ROOT_PY -c "import ${PKG}; print('  ✓ root imports', ${PKG}.__file__)"; then
    echo ""
    echo "=========================================="
    echo "Done. Restart the project task; the log should show:"
    echo "  [agent] AprilTag backend='pupil_apriltags'"
    echo "(not backend=None). Hold a tag36h11 sign in front of the camera to test."
    echo "=========================================="
else
    echo "  ✗ root still cannot import ${PKG} — see output above."
    exit 1
fi
