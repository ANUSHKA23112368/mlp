# CivicShield AI Architecture Overview

CivicShield AI unifies citizen complaint triage with cybersecurity threat operations into an integrated intelligence ecosystem.

```mermaid
graph TD
    User([Citizens / SIEM Feeds]) --> API[FastAPI Gateway :8000]
    User --> Streamlit[Streamlit 10-Tab Dashboard :8501]

    subgraph Civic Intelligence Engine
        API --> NLP[TF-IDF + NMF Topic Mining]
        NLP --> Classifiers[Logistic Regression / SVM / Random Forest]
        Classifiers --> XAI[Local Token Feature Attribution]
        Classifiers --> Sentiment[Citizen Frustration Index 0-100]
        Classifiers --> Dedup[Cosine Incident Deduplication]
    end

    subgraph Security Operations Center
        API --> RuleEngine[Deterministic Threat Rules]
        API --> Anomaly[Isolation Forest Anomaly Detection]
        API --> Correlator[Temporal IP Incident Threading]
    end

    subgraph Enterprise Controls
        API --> Auth[JWT + RBAC Admin/Analyst/Viewer]
        API --> Logging[Structlog JSON Structured Logging]
        API --> Limiter[SlowAPI Rate Limiting]
    end
```

## Core Modules
1. **CivicLens Analytics**: Automated routing, urgency escalation, SLA calculation, and geographic pressure mapping.
2. **CivicShield Security**: Event ingestion, anomaly alerts, threat clustering, and demo synthetic generator.
3. **ML Lab Pipeline**: Character + word combined n-grams, data quality assessment, and model version hashing.
