"""Tiny scripted-SSH helper for the real Duckiebot (session scratch tool).

Usage:
  .venv311\\Scripts\\python.exe _bot.py "uname -a"          # run a remote command
  .venv311\\Scripts\\python.exe _bot.py --file _bot_diag.sh  # run a script file remotely

Host/credentials come from env (set by the caller) with sane defaults:
  BOT_HOST (default 192.168.13.215), BOT_USER (auto-tries common users),
  BOT_PASS (required), BOT_PORT (22).

Any literal `__SUDO_PW__` in the command/file is replaced with BOT_PASS before
sending, so `echo '__SUDO_PW__' | sudo -S ...` works without writing the secret
to disk.
"""
import os
import sys

import paramiko

HOST = os.environ.get("BOT_HOST", "192.168.13.215")
PORT = int(os.environ.get("BOT_PORT", "22"))
PASS = os.environ.get("BOT_PASS", "")
USER_ENV = os.environ.get("BOT_USER")
USER_CANDIDATES = [USER_ENV] if USER_ENV else ["duckie", "jetson", "duckiebot", "ubuntu", "pi"]


def _read_command(argv):
    if len(argv) >= 3 and argv[1] == "--file":
        with open(argv[2], "r", encoding="utf-8") as fh:
            return fh.read()
    if len(argv) >= 2:
        return argv[1]
    return "echo connected; whoami; hostname; uname -a"


def main():
    cmd = _read_command(sys.argv).replace("__SUDO_PW__", PASS)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    last_err = None
    for user in USER_CANDIDATES:
        try:
            cli.connect(HOST, port=PORT, username=user, password=PASS,
                        timeout=15, allow_agent=False, look_for_keys=False)
            sys.stderr.write("[connected as %s@%s]\n" % (user, HOST))
            break
        except paramiko.AuthenticationException as e:
            last_err = e
            continue
        except Exception as e:
            sys.stderr.write("[connect error for %s@%s: %r]\n" % (user, HOST, e))
            sys.exit(2)
    else:
        sys.stderr.write("[auth failed for users %s: %r]\n" % (USER_CANDIDATES, last_err))
        sys.exit(2)

    exec_timeout = int(os.environ.get("BOT_TIMEOUT", "180"))
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=exec_timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    # Write via the raw buffer as UTF-8 so a cp1252 Windows console can't choke
    # on non-ASCII in remote output (build logs, progress bars, etc.).
    sys.stdout.buffer.write(out.encode("utf-8", "replace"))
    if err.strip():
        sys.stdout.buffer.write(("\n[stderr]\n" + err).encode("utf-8", "replace"))
    sys.stdout.buffer.flush()
    cli.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
