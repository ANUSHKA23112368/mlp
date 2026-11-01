# Contributing to CivicShield AI

Thank you for your interest in contributing to CivicShield AI! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/civicshield-ai.git
cd civicshield-ai

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m pytest

# Start the dashboard
streamlit run app.py

# Start the API server
uvicorn api:app --port 8000
```

## Development Guidelines

### Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Keep functions focused and under 50 lines where possible

### Testing

- All new features must include tests
- Tests go in the `tests/` directory
- Use `pytest` as the test framework
- Run the full suite before submitting: `python -m pytest -q`

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add new security correlation algorithm
fix: resolve tab definition mismatch in dashboard
docs: update API endpoint documentation
test: add tests for enhanced ML pipeline
refactor: simplify data quality checks
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all 52+ tests pass
4. Update documentation if needed
5. Submit a PR with a clear description

## Project Structure

- `src/civiclens/` — Core civic intelligence modules (analytics, NLP, ML)
- `civicshield/` — Platform extensions (auth, security, enhanced ML)
- `tests/` — Test suite
- `app.py` — Streamlit dashboard
- `api.py` — FastAPI REST service

## Reporting Issues

When reporting bugs, please include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages / stack traces

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
