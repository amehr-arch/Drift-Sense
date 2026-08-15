"""Make the repository root importable so the suite runs from a bare clone.

No packaging step, no editable install, no PYTHONPATH juggling: ``pytest`` works
immediately after ``git clone``. This matters because the submission is judged by
someone running the code on a machine we will never see.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
