"""Print desktop idle seconds; an unavailable probe must block automation.

Tried in this order:

1. GNOME's idle monitor over the session D-Bus (X11 and Wayland). A systemd
   user service has the session bus even without DISPLAY.
2. The X11 screen saver extension. DISPLAY and XAUTHORITY are taken from the
   environment, the systemd user manager (`systemctl --user show-environment`)
   or the usual locations, because a user service does not inherit them from
   the desktop. XWayland alone does not see native Wayland input, so a Wayland
   session never falls back to X11.

Other desktops can supply their own idle command in service.yaml. On failure
the reason goes to stderr as "idle probe: ..." and the exit code is 1.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PREFIX = "idle probe: "
MUTTER_DEST = "org.gnome.Mutter.IdleMonitor"
MUTTER_PATH = "/org/gnome/Mutter/IdleMonitor/Core"
MUTTER_METHOD = "org.gnome.Mutter.IdleMonitor.GetIdletime"


class ProbeError(RuntimeError):
    pass


def _run(command: list[str], env: dict[str, str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True, env=env).stdout


def manager_environment() -> dict[str, str]:
    """Variables the desktop imported into the systemd user manager."""
    try:
        output = _run(["systemctl", "--user", "show-environment"], dict(os.environ))
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def environment() -> dict[str, str]:
    env = {**manager_environment(), **os.environ}
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if "DBUS_SESSION_BUS_ADDRESS" not in env and Path(runtime, "bus").exists():
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime}/bus"
    if "DISPLAY" not in env:
        sockets = sorted(glob.glob("/tmp/.X11-unix/X*"))
        if sockets:
            env["DISPLAY"] = ":" + sockets[0].rsplit("X", 1)[-1]
    if "XAUTHORITY" not in env:
        for candidate in (Path.home() / ".Xauthority", Path(runtime, "gdm", "Xauthority")):
            if candidate.is_file():
                env["XAUTHORITY"] = str(candidate)
                break
    return env


def mutter(env: dict[str, str]) -> float | None:
    """Idle seconds from GNOME, or None if this is not a GNOME session."""
    if "DBUS_SESSION_BUS_ADDRESS" not in env:
        return None
    if shutil.which("gdbus"):
        command = [
            "gdbus", "call", "--session", "--dest", MUTTER_DEST,
            "--object-path", MUTTER_PATH, "--method", MUTTER_METHOD,
        ]  # fmt: skip
    elif shutil.which("dbus-send"):
        command = ["dbus-send", "--session", "--print-reply", f"--dest={MUTTER_DEST}", MUTTER_PATH, MUTTER_METHOD]
    else:
        return None
    try:
        output = _run(command, env)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\D*$", output.strip())
    return int(match.group(1)) / 1000 if match else None


def x11(env: dict[str, str]) -> float:
    if env.get("XDG_SESSION_TYPE") == "wayland" or "WAYLAND_DISPLAY" in env:
        raise ProbeError("Wayland session without GNOME idle monitor; configure an idle command for this desktop")
    if "DISPLAY" not in env:
        raise ProbeError("no desktop display found (DISPLAY unset, no X11 socket)")
    os.environ.update({key: env[key] for key in ("DISPLAY", "XAUTHORITY") if key in env})
    names = {name: ctypes.util.find_library(name) for name in ("X11", "Xss")}
    if not all(names.values()):
        raise ProbeError("libX11 or libXss missing (package libxss1)")
    lib = ctypes.CDLL(names["X11"])
    xss = ctypes.CDLL(names["Xss"])

    class Info(ctypes.Structure):
        _fields_ = [
            ("window", ctypes.c_ulong),
            ("state", ctypes.c_int),
            ("kind", ctypes.c_int),
            ("til_or_since", ctypes.c_ulong),
            ("idle", ctypes.c_ulong),
            ("event_mask", ctypes.c_ulong),
        ]

    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XFree.argtypes = [ctypes.c_void_p]
    xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(Info)
    xss.XScreenSaverQueryInfo.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(Info)]
    display = lib.XOpenDisplay(None)
    if not display:
        auth = "set" if "XAUTHORITY" in env else "unset"
        raise ProbeError(f"cannot open display {env['DISPLAY']} (XAUTHORITY {auth})")
    info = xss.XScreenSaverAllocInfo()
    try:
        if not info or not xss.XScreenSaverQueryInfo(display, lib.XDefaultRootWindow(display), info):
            raise ProbeError("X11 screen saver query failed")
        return info.contents.idle / 1000
    finally:
        if info:
            lib.XFree(info)
        lib.XCloseDisplay(display)


def seconds() -> float:
    env = environment()
    value = mutter(env)
    return value if value is not None else x11(env)


def main() -> int:
    try:
        print(seconds())
    except ProbeError as err:
        print(PREFIX + str(err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
