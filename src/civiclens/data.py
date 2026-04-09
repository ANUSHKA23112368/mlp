from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["complaint_text", "department", "urgency"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def demo_dataset_path() -> Path:
    return project_root() / "data" / "demo_complaints.csv"


def load_demo_data() -> pd.DataFrame:
    return load_dataset(demo_dataset_path())


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_dataset(frame)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned = frame.copy()
    cleaned["complaint_text"] = cleaned["complaint_text"].fillna("").astype(str).str.strip()
    cleaned["department"] = cleaned["department"].fillna("unknown").astype(str).str.strip().str.lower()
    cleaned["urgency"] = cleaned["urgency"].fillna("medium").astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned["complaint_text"].str.len() > 10].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Dataset has no valid complaint_text rows after cleaning.")

    return cleaned


def prepare_unlabeled_frame(texts: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"complaint_text": list(texts)})
    frame["complaint_text"] = frame["complaint_text"].fillna("").astype(str).str.strip()
    frame = frame[frame["complaint_text"].str.len() > 0].reset_index(drop=True)
    if frame.empty:
        raise ValueError("At least one complaint text is required.")
    return frame

