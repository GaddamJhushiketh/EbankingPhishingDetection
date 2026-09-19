# E-Banking Phishing Detection

A modular Flask foundation for the final-year project **Detecting E-Banking
Phishing Websites Using Associative Classification**.

Phase 1 provides the application structure and a working Flask development
server. The associative-classification model, dataset processing, feature
extraction, and persistence layers are intentionally not implemented yet.

## Project structure

```text
.
├── app.py                  # Development entry point
├── config.py               # Environment-specific configuration
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py         # Flask application factory
│   ├── routes/             # HTTP route blueprints
│   ├── services/           # Application services
│   ├── ml/                 # Reserved for future ML implementation
│   ├── models/             # Reserved for future persistence models
│   ├── templates/           # Server-rendered HTML
│   └── static/              # CSS and other static assets
├── data/
│   ├── raw/
│   └── processed/
├── models/                 # Future trained model artifacts
├── notebooks/              # Future exploratory/model notebooks
├── scripts/                # Future data and training scripts
└── tests/                  # Automated tests
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust the values if needed.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:8000/` in a browser. The health endpoint is available
at `http://127.0.0.1:8000/health`.

## Test

```bash
pytest
```

## Scope boundary

This phase does not claim any phishing-detection capability or ML results.
Model design, dataset selection, feature engineering, associative-rule
training, evaluation, and prediction workflows belong to later phases.
