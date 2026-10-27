"""
Tests for CivicShield ML Enhanced Pipeline.
Covers: data quality checks, enhanced model training, metrics computation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from civicshield.ml.enhanced_pipeline import (
    check_data_quality,
    train_enhanced_model,
    compute_enhanced_metrics,
    generate_model_version,
    hash_dataset,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "complaint_text": [
            "Water pipe burst flooding the street",
            "Electrical transformer caught fire",
            "Garbage not collected for a week",
            "Traffic signal broken at intersection",
            "Hospital emergency delay",
            "Street light not working near park",
            "Sewage overflow in residential area",
            "Road pothole causing accidents",
            "Noise complaint from construction site",
            "Public toilet in poor condition",
        ],
        "department": [
            "water", "electricity", "sanitation", "traffic",
            "health", "electricity", "water", "roads",
            "environment", "sanitation",
        ],
        "urgency": [
            "high", "critical", "medium", "high",
            "critical", "low", "high", "medium",
            "low", "low",
        ],
    })


def test_data_quality_healthy(sample_df):
    report = check_data_quality(sample_df)
    assert report.total_rows == 10
    assert report.missing_text == 0
    assert report.missing_labels == 0
    assert isinstance(report.class_distribution, dict)


def test_data_quality_detects_short_text():
    df = pd.DataFrame({
        "complaint_text": ["Hi", "OK", "Good morning and welcome to our city services"],
        "department": ["a", "b", "c"],
    })
    report = check_data_quality(df)
    assert report.very_short_texts == 2
    assert not report.is_healthy


def test_train_enhanced_model_word_only(sample_df):
    model = train_enhanced_model(
        sample_df["complaint_text"].tolist(),
        sample_df["department"].tolist(),
        use_char_features=False,
    )
    preds = model.predict(["water pipe is leaking badly"])
    assert len(preds) == 1


def test_train_enhanced_model_with_char(sample_df):
    model = train_enhanced_model(
        sample_df["complaint_text"].tolist(),
        sample_df["department"].tolist(),
        use_char_features=True,
    )
    preds = model.predict(["traffic jam near the school"])
    assert len(preds) == 1


def test_compute_enhanced_metrics():
    y_true = ["water", "electricity", "sanitation", "water"]
    y_pred = ["water", "electricity", "water", "water"]
    metrics = compute_enhanced_metrics(y_true, y_pred, label="department")
    assert "department_accuracy" in metrics
    assert "department_macro_f1" in metrics
    assert "department_weighted_f1" in metrics
    assert metrics["department_accuracy"] == 0.75


def test_model_version_format():
    version = generate_model_version("logistic_regression", "abc123def456")
    assert version.startswith("v")
    assert "logistic" in version


def test_hash_dataset_deterministic(sample_df):
    h1 = hash_dataset(sample_df)
    h2 = hash_dataset(sample_df)
    assert h1 == h2
    assert len(h1) == 64
