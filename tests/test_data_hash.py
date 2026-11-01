from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civicshield.ml.enhanced_pipeline import hash_dataset
from civiclens.data import load_demo_data


def test_hash_dataset_determinism():
    df = load_demo_data()
    h1 = hash_dataset(df)
    h2 = hash_dataset(df)
    assert h1 == h2
    assert len(h1) == 64


def test_hash_dataset_modification_changes_hash():
    df1 = load_demo_data()
    df2 = df1.copy()
    df2.iloc[0, df2.columns.get_loc("complaint_text")] = "Completely modified unique text 12345"
    h1 = hash_dataset(df1)
    h2 = hash_dataset(df2)
    assert h1 != h2
