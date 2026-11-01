# Explainable AI (XAI) Framework

CivicShield AI leverages localized token-level feature attribution to demystify complex NLP model classifications.

## Architecture

1. **TF-IDF Coefficients Mapping**:
   For linear models (Logistic Regression, Linear SVM), weights from the trained classification heads are projected directly back to input n-gram vocabularies.

2. **Token Attribution Markup**:
   Inbound text is tokenized and scored against the positive coefficient distribution for the predicted class label. Highly informative tokens receive teal background shading in the UI and high ranking scores in the REST API.

3. **Dual Head Inspection**:
   Both target heads are explained independently:
   - **Department Attribution**: Highlights tokens indicating infrastructure domain (e.g., *pothole, transformer, pipeline*).
   - **Urgency Attribution**: Highlights tokens signaling severity (e.g., *immediate, explosion, critical, fire*).
