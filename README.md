# E-Banking Phishing Detection

A modular Flask foundation for the final-year project **Detecting E-Banking
Phishing Websites Using Associative Classification**.

Phase 4 provides a reproducible three-class associative classifier in
`app/ml/associative_classifier.py`. Flask integration and live URL extraction
remain outside this phase.

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

## Phase 4 training

Install dependencies and train from the duplicate-preserving Phase 3 dataset:

```bash
python scripts/train_associative_classifier.py
```

The script uses a fixed seed (`42`) and stratified 80/20 split, mines rules
with mlxtend Apriori and `association_rules`, and writes
`models/associative_classifier.pkl`, `models/evaluation.json`, and
`models/association_rules.csv` (these are intentionally versioned
deliverables, not ignored build output). The evaluation report includes a
confusion matrix, full classification report, unmatched-test count, rule
support/confidence/lift/length characteristics, confidence-threshold
experiments, and the selected-threshold rationale. The model artifact can be loaded with
`AssociativeClassifier.load(...)`. The report distinguishes frequent itemsets,
all generated association rules, and filtered classification rules, and lists
the top ten classification rules with antecedent, consequent, support,
confidence, and lift. Matching rules are ranked by descending confidence,
then lift, support, antecedent length, class value, and lexical antecedent
order; the first rule supplies the prediction. A record with no matching rule
uses the deterministically selected majority class from the training split.

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
