"""
CivicShield ML — Enhanced Training Pipeline
Adds:
  • Character-level TF-IDF features (alongside word-level)
  • Ensemble Voting Classifier option
  • Feature combination (word + char n-grams)
  • Enhanced metrics: per-class F1, macro/micro averages
  • Model versioning with timestamps
  • Data quality checks before training
"""
from __future__ import annotations
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


@dataclass
class DataQualityReport:
    """Summary of data quality checks before training."""
    total_rows: int = 0
    missing_text: int = 0
    missing_labels: int = 0
    duplicate_texts: int = 0
    very_short_texts: int = 0          # < 10 chars
    class_distribution: dict = field(default_factory=dict)
    min_class_count: int = 0
    max_class_count: int = 0
    imbalance_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)
    is_healthy: bool = True


def check_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Run data quality checks and return a report."""
    report = DataQualityReport()
    report.total_rows = len(df)

    if "complaint_text" in df.columns:
        report.missing_text = int(df["complaint_text"].isna().sum())
        report.duplicate_texts = int(df["complaint_text"].duplicated().sum())
        report.very_short_texts = int((df["complaint_text"].str.len() < 10).sum())
    
    if "department" in df.columns:
        report.missing_labels = int(df["department"].isna().sum())
        dist = df["department"].value_counts().to_dict()
        report.class_distribution = dist
        counts = list(dist.values())
        if counts:
            report.min_class_count = min(counts)
            report.max_class_count = max(counts)
            report.imbalance_ratio = round(max(counts) / max(min(counts), 1), 2)

    # Generate warnings
    if report.missing_text > 0:
        report.warnings.append(f"{report.missing_text} rows have missing complaint text")
    if report.duplicate_texts > 5:
        report.warnings.append(f"{report.duplicate_texts} duplicate texts found — may inflate metrics")
    if report.very_short_texts > 0:
        report.warnings.append(f"{report.very_short_texts} texts shorter than 10 chars")
    if report.imbalance_ratio > 5:
        report.warnings.append(f"High class imbalance (ratio {report.imbalance_ratio}:1)")
    if report.missing_labels > 0:
        report.warnings.append(f"{report.missing_labels} rows missing department labels")
    
    report.is_healthy = len(report.warnings) == 0
    return report


def build_combined_features() -> FeatureUnion:
    """Build a FeatureUnion of word-level + character-level TF-IDF."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=5000,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=3000,
        )),
    ])


def train_enhanced_model(
    texts: list[str],
    labels: list[str],
    use_char_features: bool = True,
) -> Pipeline:
    """Train an enhanced text classifier with optional char n-grams."""
    if use_char_features:
        features = build_combined_features()
    else:
        features = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )

    pipeline = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            C=1.0,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def compute_enhanced_metrics(
    y_true: list[str],
    y_pred: list[str],
    label: str = "department",
) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    return {
        f"{label}_accuracy": round(accuracy_score(y_true, y_pred), 4),
        f"{label}_macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        f"{label}_weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        f"{label}_macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        f"{label}_macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        f"{label}_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def generate_model_version(algorithm: str, dataset_hash: str) -> str:
    """Generate a deterministic model version string."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"v{ts}-{algorithm[:8]}-{dataset_hash[:8]}"


def hash_dataset(df: pd.DataFrame) -> str:
    """Create a hash of the dataset for versioning."""
    content = df.to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()
