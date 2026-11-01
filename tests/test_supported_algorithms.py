from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.modeling import SUPPORTED_ALGORITHMS


def test_supported_algorithms_list():
    assert "logistic_regression" in SUPPORTED_ALGORITHMS
    assert len(SUPPORTED_ALGORITHMS) >= 3
    for algo in SUPPORTED_ALGORITHMS:
        assert isinstance(algo, str)
        assert len(algo) > 0
