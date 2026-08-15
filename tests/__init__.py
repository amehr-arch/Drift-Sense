"""Test suite for Drift-Sense.

Written against the standard library's ``unittest`` rather than ``pytest`` so the
suite has *zero* dependencies beyond the runtime ones. A reviewer can clone the
repository and run::

    python -m unittest discover -s tests -t .

on any Python 3.9+ installation. The tests are also plain enough that ``pytest``
collects and runs them unchanged for anyone who prefers it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
