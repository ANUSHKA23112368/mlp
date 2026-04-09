from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from civiclens.data import validate_dataset


def make_text_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
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
        return pd.DataFrame(
            {
                "complaint_text": texts,
                "predicted_department": department_predictions,
                "predicted_urgency": urgency_predictions,
            }
        )


def train_models(frame: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> TrainingArtifacts:
    data = validate_dataset(frame)

    stratify_department = data["department"] if data["department"].nunique() > 1 else None
    train_frame, test_frame = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_department,
    )

    department_model = make_text_classifier()
    urgency_model = make_text_classifier()

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

