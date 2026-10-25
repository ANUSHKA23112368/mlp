from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def discover_topics(
    texts: list[str],
    n_topics: int = 5,
    top_words: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    topics, _ = assign_topics(texts, n_topics=n_topics, top_words=top_words, random_state=random_state)
    return topics


def assign_topics(
    texts: list[str],
    n_topics: int = 5,
    top_words: int = 5,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[int]]:
    cleaned_texts = [str(t).strip() for t in texts if str(t).strip()]
    if len(cleaned_texts) < n_topics:
        return [], []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=400,
    )
    doc_topics = nmf.fit_transform(tfidf_matrix)

    # Dominant topic for each document
    dominant_topics = np.argmax(doc_topics, axis=1)

    topics_summary: list[dict[str, Any]] = []
    total_docs = len(cleaned_texts)

    for topic_idx in range(n_topics):
        # Top words for this topic
        top_feature_indices = np.argsort(nmf.components_[topic_idx])[::-1][:top_words]
        keywords = [str(feature_names[i]) for i in top_feature_indices]

        # Name is capitalized top 3 keywords
        name = " / ".join(k.replace("_", " ").title() for k in keywords[:3])

        # Documents assigned to this topic
        assigned_indices = np.where(dominant_topics == topic_idx)[0]
        sample_count = int(len(assigned_indices))
        prevalence = round((sample_count / total_docs) * 100, 1)

        # Representative documents (highest weight in this topic)
        if sample_count > 0:
            sub_weights = doc_topics[assigned_indices, topic_idx]
            top_doc_sub_indices = np.argsort(sub_weights)[::-1][:3]
            representative_complaints = [cleaned_texts[assigned_indices[i]] for i in top_doc_sub_indices]
        else:
            representative_complaints = []

        topics_summary.append({
            "topic_id": topic_idx + 1,
            "topic_name": name,
            "keywords": keywords,
            "sample_count": sample_count,
            "prevalence_pct": prevalence,
            "representative_complaints": representative_complaints,
        })

    # Sort topics by prevalence descending
    topics_summary.sort(key=lambda t: t["sample_count"], reverse=True)
    return topics_summary, dominant_topics.tolist()
