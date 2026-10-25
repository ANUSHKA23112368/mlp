from __future__ import annotations

import re
from typing import Any

FRUSTRATION_LEXICON: dict[str, int] = {
    # Emergency & Danger (weight 25-30)
    "emergency": 30,
    "danger": 28,
    "hazard": 25,
    "explod": 30,
    "explosion": 30,
    "fire": 28,
    "spark": 25,
    "collapse": 28,
    "accident": 28,
    "injur": 28,
    "poison": 30,
    "trauma": 28,
    "fatal": 30,
    # Repeated Failures & Inaction (weight 18-24)
    "repeated": 20,
    "ignored": 22,
    "unresponsive": 22,
    "no action": 24,
    "never arrived": 22,
    "three weeks": 20,
    "two weeks": 18,
    "months": 22,
    "second time": 18,
    "third time": 24,
    "multiple times": 22,
    "again": 14,
    # Anger, Disgust & Distress (weight 15-22)
    "unbearable": 22,
    "foul": 18,
    "stench": 20,
    "decomposing": 22,
    "disaster": 20,
    "harassment": 24,
    "bribe": 25,
    "unlawful": 20,
    "corrupt": 24,
    "fraud": 22,
    "illegal": 18,
    "unacceptable": 22,
    "shocking": 18,
    "pathetic": 20,
    "sick": 18,
    "illness": 18,
    "suffering": 20,
    "stranded": 20,
    # Operational Friction (weight 10-15)
    "delay": 12,
    "broken": 14,
    "leak": 14,
    "overflow": 15,
    "crash": 15,
    "offline": 15,
    "stuck": 12,
    "wrong": 12,
    "charged twice": 18,
    "refund": 14,
}

POSITIVE_LEXICON: dict[str, int] = {
    "resolved": 15,
    "fixed": 15,
    "thank": 12,
    "appreciate": 15,
    "quick": 10,
    "helpful": 12,
    "good": 8,
    "smooth": 10,
    "timely": 12,
}


def analyze_sentiment(text: str) -> dict[str, Any]:
    lowered = text.lower()
    
    # Calculate frustration raw score
    frustration_points = 0
    detected_signals: list[str] = []

    for pattern, weight in FRUSTRATION_LEXICON.items():
        if re.search(r"\b" + re.escape(pattern), lowered):
            frustration_points += weight
            detected_signals.append(pattern)

    # Exclamation mark intensity boost
    exclamations = min(lowered.count("!"), 3) * 5
    frustration_points += exclamations

    # Caps lock / shouting indicator
    uppercase_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(uppercase_words) >= 2:
        frustration_points += 15
        detected_signals.append("uppercase emphasis")

    # Positive offsets
    positive_points = 0
    for pattern, weight in POSITIVE_LEXICON.items():
        if re.search(r"\b" + re.escape(pattern), lowered):
            positive_points += weight

    # Final bounded frustration score: 0 to 100
    frustration_score = max(0, min(100, frustration_points - positive_points))
    
    # Sentiment score: -1.0 to +1.0
    sentiment_score = round(max(-1.0, min(1.0, (positive_points - frustration_points) / 50.0)), 2)

    # Escalation risk classification
    if frustration_score >= 70:
        escalation_risk = "Critical Flashpoint"
        liaison_guidance = "Assign dedicated senior liaison officer; proactive citizen outreach within 2 hours."
    elif frustration_score >= 45:
        escalation_risk = "High Risk"
        liaison_guidance = "Acknowledge frustration directly, provide trackable ticket ID, and prioritize dispatch."
    elif frustration_score >= 20:
        escalation_risk = "Elevated"
        liaison_guidance = "Standard prioritized ticketing queue with regular automated SMS updates."
    else:
        escalation_risk = "Normal"
        liaison_guidance = "Routine service desk processing with standard turnaround window."

    return {
        "sentiment_score": sentiment_score,
        "frustration_score": frustration_score,
        "escalation_risk": escalation_risk,
        "frustration_signals": detected_signals[:6],
        "liaison_guidance": liaison_guidance,
    }


def batch_analyze_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    return [analyze_sentiment(text) for text in texts]
