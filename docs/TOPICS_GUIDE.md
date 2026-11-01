# Unsupervised NLP Topic Discovery

CivicShield AI integrates Non-Negative Matrix Factorization (NMF) to discover nascent grievance themes before official department categorization exists.

## Methodology

1. **Term Matrix Construction**:
   Input text corpora are vectorized using TF-IDF with English stop words filtered out and n-gram boundaries set to `(1, 2)`.

2. **Matrix Factorization**:
   The TF-IDF document-term matrix $V$ is approximated as the product of two non-negative matrices:
   $$V \approx W \times H$$
   - $W$: Document-topic membership weights.
   - $H$: Topic-term component weights.

3. **Dynamic Topic Extraction**:
   For each discovered component, the top contributing vocabulary tokens are synthesized into a coherent topic signal and mapped to representative complaints in the queue.
