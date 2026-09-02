"""Shared pytest setup for the CorrDiff example tests.

Puts the example root (the directory holding ``train.py``, ``helpers/``,
``datasets/``, ``scripts/``) on ``sys.path`` so tests can import them
regardless of the working directory pytest was invoked from.

This replaces the ``sys.path.insert`` preamble that was copy-pasted into
individual test modules; without it, some tests worked only when pytest
happened to be run from the example root.
"""

import sys
from pathlib import Path

_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))
