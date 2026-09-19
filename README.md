# E-Banking Phishing Detection

A modular Flask foundation for the final-year project **Detecting E-Banking
Phishing Websites Using Associative Classification**.

Phase 4 provides a reproducible three-class associative classifier in
`app/ml/associative_classifier.py`. Phase 5 adds integrity-checked loading,
safe URL handling, prediction history, and SQLAlchemy persistence.

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
at `http://127.0.0.1:8000/health`; predictions are submitted at `/predict`
and prior results are shown at `/history`.

## Phase 5 configuration and safety

The application-controlled `models/associative_classifier.pkl` is hashed with
SHA-256 and compared in constant time when `MODEL_SHA256` is configured.
The model artifact is a trusted application artifact and is integrity-checked
before deserialization. Pickle files must never be accepted from untrusted
users. SHA-256 verifies the configured artifact contents; it does not make
pickle inherently safe.
Development permits a missing hash (and reports degraded health if the
artifact is missing), but production fails fast unless both `MODEL_SHA256` and
a strong environment-provided `SECRET_KEY` are present. Development generates
a random process-local secret when omitted; no predictable production fallback
exists. SQLite is the deterministic default for development/tests. Set
`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD` (or
`DATABASE_URL`) for MySQL.

URL validation permits only public HTTP(S) hosts, rejects private/link-local
DNS results, and uses bounded, revalidated redirects and response sizes.
Prediction history stores a redacted URL without credentials, query strings,
or fragments so the public history view does not disclose submitted secrets.
Page-content, traffic, WHOIS, and other non-URL-observable features must be
supplied by an explicit trusted provider; missing values are rejected rather
than fabricated. The default URL-only strategy therefore reports a controlled
error instead of making a prediction when those signals are unavailable.

## Dataset Reproducibility Limitation

This project uses the UCI Website Phishing Dataset 379. Phase 4 was trained
on its encoded feature values. UCI confirms the feature names, allowed
domains, and target encoding, but the available primary evidence does not
currently establish the feature-specific rules that convert live website
observations into those values. Consequently, `/predict` may collect raw
observations but intentionally does not convert them into model inputs or
produce a prediction. Missing, unverified, zero-defaulted, guessed, or
majority-filled values are never used.

The model remains usable through `/predict/features`, a clearly separated
development/test route for complete, already encoded Dataset 379 feature
vectors. This route does not claim that values came from the submitted URL.
A future implementation can enable live prediction if the original
feature-generation rules are recovered.

## Test

```bash
pytest
```

## Phase 4 training

Install dependencies and train from the duplicate-preserving Phase 3 dataset:

```bash
python scripts/train_associative_classifier.py
```

The script uses a fixed seed (`42`) and an approximately 80/20,
class-stratified **exact feature+target record-group split**. Identical
feature-and-target records never cross the split (`shared_exact_records=0`).
Apriori and association rules are mined on the training partition only, and
the script writes
`models/associative_classifier.pkl`, `models/evaluation.json`, and
`models/association_rules.csv` (these are intentionally versioned
deliverables, not ignored build output). The evaluation report is tagged
`evaluation_strategy=exact_record_group_aware`, records `random_seed=42`,
`test_size_target=0.20`, and includes a secondary feature-only-group analysis
for duplicate feature vectors and conflicting labels that cross splits. It
also preserves Strategy A, the historical row-level stratified split, as a
clearly labelled comparison baseline. The report includes a
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

The current rebuilt exact-record-group evaluation measures 0.7823 accuracy
and 0.6421 macro-F1 on 271 held-out records (the historical row-level
Strategy A baseline is 0.8524 accuracy and 0.6173 macro-F1). These metrics
are regenerated by the complete training command rather than hard-coded.

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
