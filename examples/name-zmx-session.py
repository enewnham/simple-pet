#!/usr/bin/env python3
"""Set a zmx title from a Simple Pet naming callback."""

import json
import subprocess
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    session = payload.get("zmx_session")
    if session:
        subprocess.run(
            [sys.argv[1], "set", session, f"title={payload['name']}"], check=True
        )


if __name__ == "__main__":
    main()
