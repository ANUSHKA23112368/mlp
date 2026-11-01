from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import format_department


def test_format_department_standard_keys():
    assert "Water" in format_department("water")
    assert "Electricity" in format_department("electricity")


def test_format_department_unknown_key():
    assert format_department("unknown_department_xyz") is not None
    assert isinstance(format_department("unknown_department_xyz"), str)
