from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']{3,}", text.lower())


def summarize_text(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = [token for token in tokenize(text) if token not in STOPWORDS]
    if not words:
        return " ".join(sentences[:max_sentences])

    frequencies = Counter(words)
    scored = []
    for sentence in sentences:
        tokens = [token for token in tokenize(sentence) if token not in STOPWORDS]
        score = sum(frequencies[token] for token in tokens) / max(len(tokens), 1)
        scored.append((score, sentence))

    best = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
    ordered = [sentence for _, sentence in best if sentence]
    return " ".join(ordered)

