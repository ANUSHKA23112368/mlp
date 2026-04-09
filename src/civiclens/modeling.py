from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from civiclens.data import validate_dataset


def make_text_classifier(label_count: int) -> Pipeline:
    classifier = (
        LogisticRegression(max_iter=2000, class_weight="balanced")
        if label_count > 1
        else DummyClassifier(strategy="most_frequent")
    )
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("clf", classifier),
        ]
    )


@dataclass
class TrainingArtifacts:
    department_model: Pipeline
    urgency_model: Pipeline
    metrics: dict[str, Any]

    def predict(self, texts: list[str]) -> pd.DataFrame:
        department_predictions = self.department_model.predict(texts)
        urgency_predictions = self.urgency_model.predict(texts)
        department_probabilities = self.department_model.predict_proba(texts)
        urgency_probabilities = self.urgency_model.predict_proba(texts)
        return pd.DataFrame(
            {
                "complaint_text": texts,
                "predicted_department": department_predictions,
                "predicted_urgency": urgency_predictions,
                "department_confidence": department_probabilities.max(axis=1).round(3),
                "urgency_confidence": urgency_probabilities.max(axis=1).round(3),
            }
        )


def safe_train_test_split(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_department = None
    if data["department"].nunique() > 1 and data["department"].value_counts().min() > 1:
        stratify_department = data["department"]

    try:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_department,
        )
    except ValueError:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def train_models(frame: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> TrainingArtifacts:
    data = validate_dataset(frame)
    train_frame, test_frame = safe_train_test_split(data, test_size=test_size, random_state=random_state)

    department_model = make_text_classifier(train_frame["department"].nunique())
    urgency_model = make_text_classifier(train_frame["urgency"].nunique())

    department_model.fit(train_frame["complaint_text"], train_frame["department"])
    urgency_model.fit(train_frame["complaint_text"], train_frame["urgency"])

    department_pred = department_model.predict(test_frame["complaint_text"])
    urgency_pred = urgency_model.predict(test_frame["complaint_text"])

    metrics = {
        "department_accuracy": round(
            accuracy_score(test_frame["department"], department_pred), 3
        ),
        "urgency_accuracy": round(accuracy_score(test_frame["urgency"], urgency_pred), 3),
        "department_report": classification_report(
            test_frame["department"],
            department_pred,
            zero_division=0,
            output_dict=True,
        ),
        "urgency_report": classification_report(
            test_frame["urgency"],
            urgency_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    return TrainingArtifacts(
        department_model=department_model,
        urgency_model=urgency_model,
        metrics=metrics,
    )
