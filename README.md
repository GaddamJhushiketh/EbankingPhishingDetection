# E-Banking Phishing Detection

A modular Flask foundation for the final-year project **Detecting E-Banking
Phishing Websites Using Associative Classification**.

Phase 4 provides a reproducible three-class associative classifier in
`app/ml/associative_classifier.py`. Phase 5 adds integrity-checked loading,
safe URL handling, prediction history, and SQLAlchemy persistence.

## Phase 6 interface

The application now includes a responsive cybersecurity-oriented interface
with Dashboard, Live Analysis, Dataset Feature Vector Test, History, and
System Status pages. Live analysis remains intentionally separate from the
feature-vector test: it may display safe raw observations, but it does not
invent Dataset 379 encodings or call the model when mappings are unavailable.
The feature-vector test accepts a complete validated Dataset 379 vector and
uses the existing Phase 4 classifier without confidence estimates.

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

## Reproducibility and ML Workflow

The authoritative input is `dataset/raw/phishing_dataset.csv`. The dataset
validator checks the exact columns, numeric domains, missing values, duplicate
rows, target distribution, and conflicting feature-only groups without
modifying the raw file:

```bash
python scripts/validate_dataset.py
```

The validator writes the measured quality report to
`models/dataset_quality.json`. The project retains duplicate rows in the
primary training copy for association-rule support calculations; the
deduplicated CSV is a separate sensitivity-analysis artifact. Conflicting
feature-only groups are reported rather than silently relabelled.

To reproduce the offline training/evaluation workflow explicitly:

```bash
python scripts/train_associative_classifier.py --data dataset/processed/phishing_clean.csv --seed 42 --test-size 0.2 --min-support 0.01 --min-confidence 0.5 --min-lift 1.0
```

This command uses the exact feature+target record-group split, mines rules
from the training partition only, and rewrites the versioned model,
evaluation, and association-rules artifacts. Run it intentionally; normal
application startup never retrains the model. The persisted evaluation
selects the validated `min_confidence=0.7` experiment on the fixed split.

`models/reproducibility_metadata.json` records actual dataset and artifact
hashes, dimensions, feature/target contracts, preprocessing information,
split configuration, and training settings. `models/model_manifest.json` is a
small registry for the existing model artifact and its evaluation summary.
These files do not alter model loading.

The offline `/predict/features` path classifies complete, already encoded
Dataset 379 vectors. Live URL analysis is a separate workflow: it may collect
raw observations, but it does not claim complete Dataset 379 encoding where
authoritative feature mappings remain unavailable.

## Model Explainability

The feature-vector result page includes model-derived evidence from the
association rules stored in the verified classifier artifact. Matching rules
retain their learned support, rule confidence, lift, consequent, and
antecedent conditions. Rules are ranked using the classifier's existing
deterministic order: descending confidence, lift, support, antecedent length,
class value, and lexical antecedent order. No new score is introduced.

Rule confidence describes how often the rule consequent occurred in the
training data for that antecedent; it is not a prediction probability and is
not presented as one. Support describes the rule's observed transaction
frequency, while lift compares the rule's consequent frequency with its
baseline frequency. The feature analysis table reports technical Dataset 379
feature names, submitted encoded values, allowed values, and matching-rule
counts. It does not assign causal or positive/negative feature importance.

Explanations are limited to learned rule evidence. They do not establish
causality, and they do not add semantic interpretations that have not been
verified for Dataset 379. Live URL analysis remains `mapping_unverified` when
the complete authoritative encoding cannot be established; it does not run
the classifier or produce rule explanations in that state.

## Security Analytics Dashboard

`/dashboard` is a read-only reporting view over persisted prediction history
and validated offline metadata. It reports total analyses, class counts,
mapping-unverified analyses, recent activity, learned-rule evidence usage,
encoded feature-value distributions, Dataset 379 evaluation metrics, dataset
quality, and safe model status information.

Dashboard filters for classification, prediction status, and date range are
validated server-side and applied through SQLAlchemy ORM predicates. No
arbitrary SQL or query expressions are accepted. Empty history and missing
optional metadata produce explicit empty states rather than fabricated zeros
or placeholder charts. The page includes print-friendly styling and states
that it is generated from application data and validated offline model
metadata.

Offline evaluation metrics describe the fixed Dataset 379 evaluation and are
not live website accuracy or a probability of correctness. Rule confidence is
rule evidence, not prediction probability. Feature analytics retain the
technical encoded values and do not invent unresolved Dataset 379 semantic
descriptions. Live URL analyses that remain `mapping_unverified` are not
counted as completed predictions.

## Final Deployment Readiness

Development setup:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

The development application uses SQLite by default, permits `MODEL_SHA256` to
be unset with an explicit unverified-loading warning, and generates a
random process-local secret when `SECRET_KEY` is omitted. Configure
`DATABASE_URL` for SQLite or the `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
and `DB_PASSWORD` variables for MySQL.

Production requires both a strong environment-provided `SECRET_KEY` and the
exact SHA-256 `MODEL_SHA256` for the trusted
`models/associative_classifier.pkl` artifact. Generate a secret with
`python -c "import secrets; print(secrets.token_urlsafe(32))"` and never
commit it. Run the application with a production WSGI server rather than
Flask's development server.

Run the complete deterministic test suite with `pytest -q tests`. The
reproducibility checks and metadata are maintained by
`scripts/validate_dataset.py`, `models/dataset_quality.json`,
`models/reproducibility_metadata.json`, and `models/model_manifest.json`.
The read-only security analytics dashboard is available at `/dashboard`.

The trusted prediction workflow is `/predict/features`, which accepts a
complete already encoded Dataset 379 vector. Live URL analysis remains
mapping-unavailable when authoritative feature-generation rules cannot be
verified; it must not be described as a fully trained end-to-end live
website predictor. URL fetching remains restricted to public HTTP(S)
destinations with bounded, revalidated requests. Pickle is a trusted
application artifact format; integrity checking does not make untrusted
pickle safe.
