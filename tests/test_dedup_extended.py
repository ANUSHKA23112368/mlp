from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.deduplication import find_similar_complaints


def test_similar_complaints_empty_list():
    matches = find_similar_complaints(
        query_text="Pipeline leak in ward 3",
        database_texts=[],
        top_k=3,
        threshold=0.2,
    )
    assert matches == []


def test_similar_complaints_no_matches_above_threshold():
    texts = [
        "Noise complaint regarding loud late night pub music.",
        "Stray dog menace near the community garden park.",
    ]
    matches = find_similar_complaints(
        query_text="Major water pipeline burst flooding basement",
        database_texts=texts,
        top_k=3,
        threshold=0.85,
    )
    assert matches == []
