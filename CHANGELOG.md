# Changelog

All notable changes to the CivicShield AI platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-11-01

### Added
- **10-Tab Dashboard**: Complete interactive Streamlit UI containing Command Center, Case Studio with XAI, Incident Clusters, Emerging Topics (NMF), Data Explorer, Ops Console, Model Benchmark, Security Operations Center, ML Lab, and Data Quality Dashboard.
- **Security Operations Center**: Synthetic demo event generator, deterministic rule-based threat engine, Isolation Forest ML anomaly detector, and temporal IP correlation.
- **ML Lab Pipeline**: Word + Character n-gram FeatureUnion for improved typo and morphology handling, data quality reporting, and dataset hash versioning.
- **REST API Extensions**: `/security/scan` and `/security/summary` endpoints added to FastAPI gateway.
- **Kubernetes Deployments**: Complete Kubernetes manifests for API and Dashboard with health probes and ConfigMap.
- **CI/CD Quality**: Automated Flake8 linting workflow, GitHub bug report, feature request, and pull request templates.
- **Documentation**: Comprehensive architecture diagrams, API reference manual, contributing guide, and MIT license.

### Fixed
- Fixed Streamlit `st.tabs` unpacking issue ensuring all 10 tabs instantiate reliably.
- Updated API `/health` version to 3.0.0 with synchronised integration tests.
