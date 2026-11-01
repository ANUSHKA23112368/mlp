# Incident Deduplication Engine

During major urban disruptions (water main bursts, power substation outages, bridge flooding), citizens lodge dozens of duplicate tickets regarding the same physical incident.

## Pipeline

1. **TF-IDF Vectorization**:
   Narratives are transformed into sparse term frequency-inverse document frequency embeddings.

2. **Cosine Similarity Matrix**:
   Pairwise similarity scores between tickets are computed using the dot product of normalized term vectors:
   $$\text{similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

3. **Clustering & Corroboration**:
   Reports exceeding the cosine threshold (default 0.22 - 0.24) are grouped into incident clusters, designating the earliest complaint as the primary dispatch and grouping subsequent calls as corroborating reports.

4. **Resource Savings Metric**:
   Redundant crew dispatches avoided is quantified directly:
   $$\text{Dispatches Saved} = \text{Total Reports} - \text{Unique Physical Clusters}$$
