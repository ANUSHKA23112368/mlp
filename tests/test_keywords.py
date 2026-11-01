from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.analytics import top_keywords


def test_top_keywords_extraction():
    texts = [
        "Water main burst and flooding road.",
        "Water pipeline leaking heavily near apartments.",
        "Clean drinking water contaminated.",
    ]
    kws = top_keywords(texts, limit=3)
    assert len(kws) <= 3
    words = [k[0] for k in kws]
    assert "water" in words


def test_top_keywords_empty():
    kws = top_keywords([], limit=5)
    assert kws == []
