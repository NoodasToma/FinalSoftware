set +e
echo "=== identity ==="
whoami; id
hostname
echo
echo "=== python3 ==="
which python3; python3 --version 2>&1
echo
echo "=== how is the task launched? (dashboard / service) ==="
ps -ef | grep -iE "dashboard|deploy|real_server|flask|8000" | grep -v grep
echo "--- systemd units mentioning duck/dash ---"
systemctl list-units --type=service 2>/dev/null | grep -iE "duck|dash|bot" | head
echo
echo "=== apriltag libs (LOGIN user $(whoami)) ==="
python3 - <<'PY' 2>&1
for m in ["pupil_apriltags","dt_apriltags"]:
    try:
        mod=__import__(m); print("  OK  ",m,getattr(mod,"__version__","?"))
    except Exception as e:
        print("  MISS",m,"->",type(e).__name__,str(e)[:100])
PY
echo "--- pip3 list (apriltag/numpy/cv2) login user ---"
python3 -m pip list 2>/dev/null | grep -iE "april|numpy|opencv"
echo
echo "=== apriltag libs (ROOT, since task may run as root) ==="
echo '__SUDO_PW__' | sudo -S -p "" python3 - <<'PY' 2>&1
for m in ["pupil_apriltags","dt_apriltags"]:
    try:
        mod=__import__(m); print("  OK  ",m,getattr(mod,"__version__","?"))
    except Exception as e:
        print("  MISS",m,"->",type(e).__name__,str(e)[:100])
PY
echo "--- root pip3 list (apriltag) ---"
echo '__SUDO_PW__' | sudo -S -p "" python3 -m pip list 2>/dev/null | grep -iE "april"
echo
echo "=== apriltags install script present on bot? ==="
ls -la /home/ente/DuckieTown-Rewritten/duckiebot/apriltags/ 2>/dev/null || echo "(no apriltags dir under deployed repo)"
echo
echo "=== pip availability ==="
python3 -m pip --version 2>&1
echo '__SUDO_PW__' | sudo -S -p "" python3 -m pip --version 2>&1
echo "=== done ==="
