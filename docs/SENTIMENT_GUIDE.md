# Citizen Frustration & Escalation Risk Index

CivicShield AI features a calibrated sentiment and distress scoring heuristic designed specifically for public grievance triage.

## Metric Components

1. **Urgency Lexicon Weighting**:
   Explicit stress keywords (*emergency, collapsed, electrocution, death, fire*) heavily increment the frustration index.

2. **Repetition Signals**:
   Phrases signaling persistent municipal neglect (*"third time calling"*, *"nobody came"*, *"for weeks"*) escalate tickets from standard operational priority to dedicated liaison oversight.

3. **Escalation Classifications**:
   - `0 - 39`: **Normal** — Standard automated SLA queue.
   - `40 - 64`: **Elevated** — Supervisor flagged.
   - `65 - 84`: **High Risk** — Expedited resolution requested.
   - `85 - 100`: **Critical Flashpoint** — Immediate emergency liaison dispatch.
