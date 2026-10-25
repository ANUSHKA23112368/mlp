from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from civiclens.data import validate_dataset

SUPPORTED_ALGORITHMS = ["logistic_regression", "linear_svm", "multinomial_nb", "random_forest"]


def get_base_classifier(algorithm: str) -> Any:
    algo = algorithm.lower().strip()
    if algo == "linear_svm":
        return CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=3000),
            cv=3,
        )
    if algo == "multinomial_nb":
        return MultinomialNB(alpha=0.3)
    if algo == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    # Default: logistic_regression
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)


def make_text_classifier(label_count: int, algorithm: str = "logistic_regression") -> Pipeline:
    classifier = (
        get_base_classifier(algorithm)
        if label_count > 1
        else DummyClassifier(strategy="most_frequent")
    )
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


@dataclass
class TrainingArtifacts:
    department_model: Pipeline
    urgency_model: Pipeline
    metrics: dict[str, Any]
    algorithm: str = "logistic_regression"

    def predict(self, texts: list[str]) -> pd.DataFrame:
        clean_texts = [str(t) for t in texts]
        department_predictions = self.department_model.predict(clean_texts)
        urgency_predictions = self.urgency_model.predict(clean_texts)

        if hasattr(self.department_model, "predict_proba"):
            department_probabilities = self.department_model.predict_proba(clean_texts).max(axis=1)
        else:
            department_probabilities = np.ones(len(clean_texts))

        if hasattr(self.urgency_model, "predict_proba"):
            urgency_probabilities = self.urgency_model.predict_proba(clean_texts).max(axis=1)
        else:
            urgency_probabilities = np.ones(len(clean_texts))

        return pd.DataFrame(
            {
                "complaint_text": clean_texts,
                "predicted_department": department_predictions,
                "predicted_urgency": urgency_predictions,
                "department_confidence": department_probabilities.round(3),
                "urgency_confidence": urgency_probabilities.round(3),
            }
        )


def safe_train_test_split(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

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


def train_models(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    algorithm: str = "logistic_regression",
) -> TrainingArtifacts:
    data = validate_dataset(frame)
    train_frame, test_frame = safe_train_test_split(data, test_size=test_size, random_state=random_state)

    department_model = make_text_classifier(train_frame["department"].nunique(), algorithm=algorithm)
    urgency_model = make_text_classifier(train_frame["urgency"].nunique(), algorithm=algorithm)

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
        "algorithm": algorithm,
        "sample_count": len(frame),
        "train_count": len(train_frame),
        "test_count": len(test_frame),
    }

    # Add confusion matrices
    dept_labels = sorted(data["department"].unique().tolist())
    urg_labels = sorted(data["urgency"].unique().tolist())

    metrics["confusion_matrices"] = {
        "department": {
            "labels": dept_labels,
            "matrix": confusion_matrix(test_frame["department"], department_pred, labels=dept_labels).tolist(),
        },
        "urgency": {
            "labels": urg_labels,
            "matrix": confusion_matrix(test_frame["urgency"], urgency_pred, labels=urg_labels).tolist(),
        },
    }

    return TrainingArtifacts(
        department_model=department_model,
        urgency_model=urgency_model,
        metrics=metrics,
        algorithm=algorithm,
    )


def benchmark_models(
    frame: pd.DataFrame, cv_splits: int = 3, random_state: int = 42
) -> dict[str, Any]:
    data = validate_dataset(frame)
    texts = data["complaint_text"].values
    dept_labels = data["department"].values
    urg_labels = data["urgency"].values

    results: list[dict[str, Any]] = []

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for algo in SUPPORTED_ALGORITHMS:
        start_time = time.perf_counter()

        dept_acc_scores = []
        dept_f1_scores = []
        urg_acc_scores = []
        urg_f1_scores = []

        for train_idx, val_idx in skf.split(texts, dept_labels):
            x_train, x_val = texts[train_idx], texts[val_idx]
            y_train_d, y_val_d = dept_labels[train_idx], dept_labels[val_idx]
            y_train_u, y_val_u = urg_labels[train_idx], urg_labels[val_idx]

            model_d = make_text_classifier(len(np.unique(y_train_d)), algorithm=algo)
            model_d.fit(x_train, y_train_d)
            pred_d = model_d.predict(x_val)
            dept_acc_scores.append(accuracy_score(y_val_d, pred_d))
            dept_f1_scores.append(f1_score(y_val_d, pred_d, average="weighted", zero_division=0))

            model_u = make_text_classifier(len(np.unique(y_train_u)), algorithm=algo)
            model_u.fit(x_train, y_train_u)
            pred_u = model_u.predict(x_val)
            urg_acc_scores.append(accuracy_score(y_val_u, pred_u))
            urg_f1_scores.append(f1_score(y_val_u, pred_u, average="weighted", zero_division=0))

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        results.append({
            "algorithm": algo,
            "display_name": algo.replace("_", " ").title(),
            "department_accuracy": round(float(np.mean(dept_acc_scores)), 3),
            "department_weighted_f1": round(float(np.mean(dept_f1_scores)), 3),
            "urgency_accuracy": round(float(np.mean(urg_acc_scores)), 3),
            "urgency_weighted_f1": round(float(np.mean(urg_f1_scores)), 3),
            "training_latency_ms": elapsed_ms,
        })

    # Sort by department weighted F1 descending
    results.sort(key=lambda r: r["department_weighted_f1"], reverse=True)
    best_algo = results[0]["algorithm"]

    return {
        "leaderboard": results,
        "best_algorithm": best_algo,
        "dataset_rows": len(data),
        "cv_folds": cv_splits,
    }


def save_artifacts(artifacts: TrainingArtifacts, directory: str | Path) -> Path:
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.department_model, dir_path / "department_model.joblib")
    joblib.dump(artifacts.urgency_model, dir_path / "urgency_model.joblib")
    joblib.dump(artifacts.metrics, dir_path / "metrics.joblib")
    joblib.dump({"algorithm": artifacts.algorithm}, dir_path / "metadata.joblib")
    return dir_path


def load_artifacts(directory: str | Path) -> TrainingArtifacts:
    dir_path = Path(directory)
    dept_model = joblib.load(dir_path / "department_model.joblib")
    urg_model = joblib.load(dir_path / "urgency_model.joblib")
    metrics = joblib.load(dir_path / "metrics.joblib")
    algo = "logistic_regression"
    if (dir_path / "metadata.joblib").exists():
        algo = joblib.load(dir_path / "metadata.joblib").get("algorithm", "logistic_regression")
    return TrainingArtifacts(
        department_model=dept_model,
        urgency_model=urg_model,
        metrics=metrics,
        algorithm=algo,
    )
