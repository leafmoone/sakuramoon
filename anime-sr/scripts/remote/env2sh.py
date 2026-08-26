#!/usr/bin/env python
"""Convert the NUL-separated env dump (mig-workload-env) to a shell-sourceable file.

The G1 migration env file on sakrua2 is a raw NUL-separated `env` dump
(0 newlines). This rewrites it as quoted `export VAR=...` lines so that
`set -a; . /root/anime-sr-env; set +a` works in non-interactive shells.
"""

import shlex
import sys


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    n = 0
    with open(dst, "w") as f:
        for p in data.split(b"\0"):
            if b"=" not in p:
                continue
            key, _, val = p.decode("utf-8", "replace").partition("=")
            if key and all(c.isalnum() or c == "_" for c in key):
                f.write(f"export {key}={shlex.quote(val)}\n")
                n += 1
    print(f"wrote {n} vars to {dst}")


if __name__ == "__main__":
    main()
