from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.data import load_demo_data
from civiclens.modeling import (
    SUPPORTED_ALGORITHMS,
    benchmark_models,
    load_artifacts,
    save_artifacts,
    train_models,
)


def test_train_models_all_algorithms():
    df = load_demo_data()
    for algo in SUPPORTED_ALGORITHMS:
        artifacts = train_models(df, algorithm=algo)
        pred = artifacts.predict(["Water pipeline broke and caused leak"])
        assert len(pred) == 1
        assert pred["department_confidence"].between(0, 1).all()
        assert artifacts.metrics["algorithm"] == algo
        assert "confusion_matrices" in artifacts.metrics


def test_model_persistence():
    df = load_demo_data()
    artifacts = train_models(df, algorithm="linear_svm")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_artifacts(artifacts, tmp_dir)
        loaded = load_artifacts(tmp_dir)
        
        test_text = ["Power surge damaged appliances in our building."]
        pred_orig = artifacts.predict(test_text)
        pred_loaded = loaded.predict(test_text)
        
        assert pred_orig["predicted_department"].iloc[0] == pred_loaded["predicted_department"].iloc[0]
        assert pred_orig["predicted_urgency"].iloc[0] == pred_loaded["predicted_urgency"].iloc[0]


def test_benchmark_models():
    df = load_demo_data()
    results = benchmark_models(df, cv_splits=3)
    assert "leaderboard" in results
    assert len(results["leaderboard"]) == len(SUPPORTED_ALGORITHMS)
    assert results["best_algorithm"] in SUPPORTED_ALGORITHMS
    assert results["cv_folds"] == 3
