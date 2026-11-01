from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.data import validate_dataset, load_demo_data
from civiclens.sentiment import analyze_sentiment


def test_validate_dataset_missing_columns():
    df = pd.DataFrame({"some_random_column": [1, 2, 3]})
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_validate_dataset_empty_dataframe():
    df = pd.DataFrame(columns=["complaint_text", "department", "urgency"])
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_sentiment_empty_string():
    res = analyze_sentiment("")
    assert "frustration_score" in res
    assert res["frustration_score"] == 0 or isinstance(res["frustration_score"], int)


def test_sentiment_whitespace_only():
    res = analyze_sentiment("     \n\t  ")
    assert "frustration_score" in res
    assert isinstance(res["frustration_score"], int)
