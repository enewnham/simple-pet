import ctypes
import os
import sys


def main() -> int:
    sandbox = ctypes.CDLL("/usr/lib/system/libsystem_sandbox.dylib")
    sandbox.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    sandbox.sandbox_check.restype = ctypes.c_int
    if sandbox.sandbox_check(os.getpid(), None, 0) != 0:
        print(
            "Desktop access requires running outside the agent sandbox. "
            "Rerun the make command with sandbox restrictions disabled.",
            file=sys.stderr,
        )
        return 1

    from AppKit import NSApplication, NSScreen  # ty: ignore[unresolved-import]

    NSApplication.sharedApplication()
    print(int(bool(NSScreen.screens())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
