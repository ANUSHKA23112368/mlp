from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.summarizer import summarize_text


def test_summarize_empty_string():
    res = summarize_text("")
    assert res == "" or isinstance(res, str)


def test_summarize_single_sentence():
    sentence = "A dangerous electric cable is hanging loosely over the main footpath."
    res = summarize_text(sentence, max_sentences=1)
    assert len(res) > 0
    assert "dangerous" in res


def test_summarize_multisentence_truncation():
    text = (
        "Streetlights have failed across ward 5. "
        "Residents are afraid to walk at night due to dark alleys. "
        "Multiple reports have been submitted to the municipal corporation without resolution. "
        "Immediate repair work is requested."
    )
    res = summarize_text(text, max_sentences=2)
    assert len(res) > 0
    assert res.count(".") <= 3
