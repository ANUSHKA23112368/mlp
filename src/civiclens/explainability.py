from __future__ import annotations

import html
import re
from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline


def _get_classifier_weights(classifier: Any, class_idx: int) -> np.ndarray | None:
    # 1. Direct linear models (LogisticRegression, LinearSVC, SGDClassifier)
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
        if coef.ndim == 1:
            # Binary classification
            return coef if class_idx == 1 else -coef
        return coef[class_idx]

    # 2. CalibratedClassifierCV
    if hasattr(classifier, "calibrated_classifiers_"):
        coefs = []
        for cc in classifier.calibrated_classifiers_:
            estimator = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                c = estimator.coef_
                if c.ndim == 1:
                    coefs.append(c if class_idx == 1 else -c)
                else:
                    coefs.append(c[class_idx])
        if coefs:
            return np.mean(coefs, axis=0)

    # 3. MultinomialNB
    if hasattr(classifier, "feature_log_prob_"):
        log_probs = classifier.feature_log_prob_
        mean_log_prob = np.mean(log_probs, axis=0)
        return log_probs[class_idx] - mean_log_prob

    # 4. Tree-based models (RandomForestClassifier, ExtraTreesClassifier)
    if hasattr(classifier, "feature_importances_"):
        return classifier.feature_importances_

    return None


def explain_prediction(
    pipeline: Pipeline,
    text: str,
    target_class: str | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    # Transform text to sparse vector
    tfidf_vec = vectorizer.transform([text]).toarray()[0]
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get predictions & classes
    classes = list(classifier.classes_)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
    else:
        pred = pipeline.predict([text])[0]
        predicted_idx = classes.index(pred)
        confidence = 1.0

    if target_class and target_class in classes:
        target_idx = classes.index(target_class)
    else:
        target_idx = predicted_idx

    target_label = classes[target_idx]

    # Active features in the document
    active_mask = tfidf_vec > 0
    if not np.any(active_mask):
        return {
            "target_class": target_label,
            "confidence": round(confidence, 3),
            "top_features": [],
            "positive_features": [],
            "negative_features": [],
            "highlighted_html": html.escape(text),
        }

    weights = _get_classifier_weights(classifier, target_idx)
    if weights is not None:
        contributions = tfidf_vec * weights
    else:
        contributions = tfidf_vec

    active_indices = np.where(active_mask)[0]
    active_tokens = feature_names[active_indices]
    active_scores = contributions[active_indices]

    # Sort contributions
    sorted_order = np.argsort(active_scores)[::-1]
    sorted_tokens = active_tokens[sorted_order]
    sorted_scores = active_scores[sorted_order]

    positive_features = [
        {"token": str(token), "score": round(float(score), 4), "impact": "positive"}
        for token, score in zip(sorted_tokens, sorted_scores)
        if score > 0
    ][:top_k]

    negative_features = [
        {"token": str(token), "score": round(float(score), 4), "impact": "negative"}
        for token, score in zip(sorted_tokens[::-1], sorted_scores[::-1])
        if score < 0
    ][:top_k]

    all_features = positive_features + negative_features

    # Build highlighted HTML
    highlighted_html = _generate_highlighted_text(text, positive_features)

    return {
        "target_class": target_label,
        "confidence": round(confidence, 3),
        "top_features": all_features,
        "positive_features": positive_features,
        "negative_features": negative_features,
        "highlighted_html": highlighted_html,
    }


def _generate_highlighted_text(text: str, positive_features: list[dict[str, Any]]) -> str:
    escaped = html.escape(text)
    if not positive_features:
        return escaped

    max_score = max((f["score"] for f in positive_features), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    # Sort tokens by length descending so longer ngrams match first
    tokens = sorted(positive_features, key=lambda f: len(f["token"]), reverse=True)

    for item in tokens:
        token = item["token"]
        score = item["score"]
        opacity = min(0.85, max(0.2, score / max_score * 0.75 + 0.15))
        style = f"background-color: rgba(15, 139, 141, {opacity:.2f}); padding: 2px 5px; border-radius: 4px; font-weight: 600; color: #042b2c;"
        # Case-insensitive replacement
        pattern = re.compile(rf"\b({re.escape(token)})\b", re.IGNORECASE)
        escaped = pattern.sub(rf'<mark style="{style}">\1</mark>', escaped)

    return escaped
