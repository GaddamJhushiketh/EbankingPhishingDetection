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

Phase 1 does not claim any phishing-detection capability or ML results.
Dataset analysis and reusable feature/item preprocessing are now available,
but associative-rule training, evaluation, and prediction workflows belong to
later phases.

## Phase 3 preprocessing contract

The reusable preprocessing module is `app/ml/preprocessing.py`. It validates
the nine observed feature columns and the `Result` target using the UCI domain
`{-1, 0, 1}`. The target remains three-class and is represented as:

```text
-1 -> Result=Phishy
 0 -> Result=Suspicious
 1 -> Result=Legitimate
```

Feature items preserve both the feature name and value, for example
`SFH=1`, `SSLfinal_State=-1`, and `web_traffic=0`. The duplicate-preserving
validated input is `dataset/processed/phishing_clean.csv`; the encoded
association-mining artifact is
`dataset/processed/associative_mining_dataset.csv`. The earlier
`phishing_dataset_deduplicated.csv` remains a separate sensitivity-analysis
artifact.
