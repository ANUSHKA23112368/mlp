from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civiclens.data import load_demo_data
from civiclens.topics import assign_topics, discover_topics


def test_discover_topics():
    df = load_demo_data()
    topics = discover_topics(df["complaint_text"].tolist(), n_topics=4)
    assert len(topics) == 4
    for t in topics:
        assert "topic_name" in t
        assert len(t["keywords"]) > 0
        assert t["prevalence_pct"] > 0
        assert t["sample_count"] > 0


def test_assign_topics():
    texts = [
        "Major water pipeline burst flooding roadway.",
        "Transformer caught fire and blacked out power grid.",
        "School buses skipped the stop and students stranded.",
        "Citizen portal error 500 when paying tax.",
    ]
    topics, assignments = assign_topics(texts, n_topics=2)
    assert len(topics) == 2
    assert len(assignments) == len(texts)
    assert all(a in [0, 1] for a in assignments)
