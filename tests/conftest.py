"""Put the repository root and this directory on sys.path.

The root so tests can import `catan` and the legacy flat modules; this directory so they
can import the shared `helpers` module.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
