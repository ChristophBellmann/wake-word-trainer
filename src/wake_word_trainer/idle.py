"""Print desktop idle seconds on X11; an unavailable probe must block automation.

Other desktops can supply their own idle command in service.yaml. XWayland
alone does not measure native Wayland input, so this probe refuses Wayland.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os


def seconds() -> float:
    if os.environ.get("XDG_SESSION_TYPE") != "x11":
        raise RuntimeError("X11 session required; configure an idle probe for this desktop")
    x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
    xss = ctypes.CDLL(ctypes.util.find_library("Xss"))

    class Info(ctypes.Structure):
        _fields_ = [
            ("window", ctypes.c_ulong),
            ("state", ctypes.c_int),
            ("kind", ctypes.c_int),
            ("til_or_since", ctypes.c_ulong),
            ("idle", ctypes.c_ulong),
            ("event_mask", ctypes.c_ulong),
        ]

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XFree.argtypes = [ctypes.c_void_p]
    xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(Info)
    xss.XScreenSaverQueryInfo.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(Info)]
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Desktop display unavailable")
    info = xss.XScreenSaverAllocInfo()
    try:
        if not info or not xss.XScreenSaverQueryInfo(display, x11.XDefaultRootWindow(display), info):
            raise RuntimeError("Desktop idle query failed")
        return info.contents.idle / 1000
    finally:
        if info:
            x11.XFree(info)
        x11.XCloseDisplay(display)


if __name__ == "__main__":
    print(seconds())
