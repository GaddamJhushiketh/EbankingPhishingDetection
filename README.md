# E-Banking Phishing Detection

## Project overview

**Detecting E-Banking Phishing Websites Using Associative Classification** is
a Flask application that applies a reproducible three-class associative
classifier to encoded records from the UCI Website Phishing Dataset, Dataset
379. It provides an offline feature-vector testing workflow, model-derived
association-rule evidence, prediction history, a security analytics
dashboard, reproducibility metadata, and a security-controlled live URL
observation workflow.

The application does not claim a complete live-web predictor for ordinary
websites. Live URL analysis now applies only the evidence-backed Dataset 379
rules that can be collected from the page: URL length, external resource
percentage, external anchor percentage, form-handler (SFH), popup credential
fields, and hostname IP status. Trusted domain-registration age is supported
when supplied by an available provider. Empty and `javascript:` anchors remain
in the anchor denominator but are not external; resources count `img`,
`script`, `link`, `video`, and `audio` references. Popup JavaScript without
credential/input fields is not treated as phishing. SSL categorical state and
the historical web-traffic provider/cutoff remain `mapping_unverified`, and
domain age is also unresolved without trustworthy registration data. If any
of the nine values is unresolved, live analysis reports
`feature_mapping_unavailable` and does not call the classifier.

## Problem statement and objectives

Phishing websites imitate legitimate online banking services to capture
credentials and other sensitive information. The project investigates whether
association rules can provide a transparent three-class classification
workflow for phishing, suspicious, and legitimate records while maintaining
reproducibility and safe handling of submitted URLs.

The objectives are to validate the Dataset 379 contract, mine deterministic
associative classification rules, evaluate them without exact-record leakage,
provide model-derived rule evidence, persist safe prediction history, expose
security controls for URL analysis, and prepare an auditable demonstration and
reporting workflow.

## Technology stack

Python, Flask, Jinja templates, SQLAlchemy/Flask-SQLAlchemy, SQLite for local
development and tests, optional MySQL through PyMySQL, Requests, Beautiful
Soup, Pandas, NumPy, mlxtend, Matplotlib/Seaborn for EDA, pytest, and
Notebook/nbconvert are used by the repository.

## Final implementation inventory

| Area | Implemented responsibility |
|---|---|
| Flask application | Application factory, configuration selection, extensions, startup |
| Configuration | Development, testing, and production settings; environment variables |
| Database/history | `PredictionHistory`, additive legacy-schema compatibility, SQLAlchemy persistence |
| Dataset handling | Raw/processed Dataset 379 files, validation, EDA, quality metadata |
| Associative classifier | Apriori mining, class-rule filtering, prediction, deterministic fallback |
| Rule generation | Support, confidence, lift, class consequents, persisted audit CSV |
| Feature validation | Exact nine-feature contract, per-feature domains, target mapping |
| Live URL analysis | Safe raw observations and explicit mapping-unavailable state |
| URL security | URL validation, DNS checks, SSRF blocking, redirects, bounded HTTP, redaction |
| Prediction service | Verified model loading, encoded-vector prediction, history orchestration |
| Explainability | Matching rule evidence and deterministic ranking |
| Analytics dashboard | Counts, activity, feature distributions, evaluation, quality, status, filters |
| Reproducibility | Dataset validation, hashes, manifests, metadata, deterministic training settings |
| Integrity verification | SHA-256 and constant-time comparison before trusted model loading |
| UI/templates | Home, live analysis, feature testing, results, history, dashboard, status |
| Testing | Application, preprocessing, model, security, UI, dashboard, and hardening regression tests |
| Deployment configuration | `.env.example`, production secret/hash requirements, SQLite/MySQL settings |

## Implemented capabilities

- Flask web interface and application factory.
- Strict Dataset 379 feature-vector validation.
- Apriori association-rule mining and deterministic classification.
- Exact feature-plus-target record-group-aware offline evaluation.
- SHA-256 verification before trusted model deserialization.
- SSRF-safe, bounded, TLS-verified URL observation.
- Prediction history through SQLAlchemy with SQLite and MySQL-compatible
  configuration.
- Explainability through matching learned rule evidence.
- Read-only security analytics dashboard with validated filters.
- Dataset quality, model manifest, evaluation, and reproducibility metadata.
- Responsive pages for home, live analysis, feature-vector testing, results,
  history, dashboard, and system status.

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
│   ├── __init__.py         # Flask application factory and extensions
│   ├── routes/             # HTTP routes and request handling
│   ├── services/           # Prediction, extraction, security, explanation
│   ├── ml/                 # Dataset contract and associative classifier
│   ├── models/             # SQLAlchemy prediction-history model
│   ├── templates/          # Server-rendered HTML
│   └── static/             # Responsive and print-friendly CSS
├── dataset/
│   ├── raw/                # Protected UCI-derived input
│   └── processed/          # Validated and mining representations
├── models/                 # Trusted model and audit metadata
├── notebooks/              # Dataset EDA notebook
├── scripts/                # Validation, metadata, and training workflows
├── docs/                   # Viva, report, presentation, and demo material
└── tests/                  # Automated regression and security tests
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
on its encoded feature values. The operational live rules above are based on
published reproductions of the Dataset 379 feature methodology, not on a
claim that the original data-collection implementation has been recovered.
Consequently, `/predict` converts only verified observations and does not
produce a prediction when any required mapping is unresolved. Missing,
unverified, zero-defaulted, guessed, or
majority-filled values are never used.

The model remains usable through `/predict/features`, a clearly separated
development/test route for complete, already encoded Dataset 379 feature
vectors. This route does not claim that values came from the submitted URL.
The complete nine-feature vector can therefore be produced only when a
trusted source supplies the remaining provider-dependent observations. The
offline evaluation metrics below are not live-web accuracy.

## Tests

```bash
pytest -q tests
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

## Dataset 379

The authoritative dataset is the UCI Website Phishing Dataset, Dataset 379.
The repository's protected raw file contains 1,353 rows and 10 columns: nine
features and the `Result` target. The exact feature contract is:

```text
SFH, popUpWidnow, SSLfinal_State, Request_URL, URL_of_Anchor,
web_traffic, URL_Length, age_of_domain, having_IP_Address
```

The target mapping is `-1 = Phishy`, `0 = Suspicious`, and
`1 = Legitimate`. The verified target counts are 702, 103, and 548
respectively. The dataset has no missing values, 629 exact duplicate rows,
724 unique rows after exact deduplication, and 48 conflicting feature-only
groups. Duplicate rows are retained in the primary training representation;
the deduplicated file is a separate sensitivity-analysis artifact.

The project preserves the verified feature domains and target mapping. It
does not claim undocumented meanings for individual `-1`, `0`, and `1`
feature values, and it does not invent operational live-extraction
thresholds.

## Associative classification methodology and evaluation

The preprocessing contract validates the nine feature columns and target,
then represents each row as feature-value items plus a class item. Apriori
mines frequent itemsets on the training partition. Rules are retained when
they have a class consequent and satisfy the configured support, confidence,
and lift constraints. The validated workflow uses `min_support=0.01`,
`min_confidence=0.70`, `min_lift=1.0`, and seed `42`. The training script's
command-line default confidence argument is `0.5`; the persisted evaluation
selects and records the validated `0.7` experiment.

Matching rules are ordered by descending confidence, lift, support,
antecedent length, class value, and lexical antecedent order. The first
match supplies the class; if no rule matches, the deterministic training
majority class is used. The same learned matches supply the result-page
explanation.

The primary result is an offline Dataset 379 evaluation using an exact
feature-plus-target record-group-aware split: 1,082 training records and
271 test records, with no shared exact records. It must not be described as
live-web detection accuracy:

| Metric | Value |
|---|---:|
| Accuracy | 0.7822878228782287 |
| Macro precision | 0.8522692601067887 |
| Macro recall | 0.6165223665223665 |
| Macro F1 | 0.6420598680872653 |
| Weighted F1 | 0.7669808533701776 |

Per-class results are Phishy: precision 0.7829, recall 0.8500, F1 0.8151,
support 140; Suspicious: precision 1.0000, recall 0.1905, F1 0.3200,
support 21; and Legitimate: precision 0.7739, recall 0.8091, F1 0.7911,
support 110. The confusion matrix, in class order `-1, 0, 1`, is:

```text
Actual -1: 119 0 21
Actual  0:  12 4  5
Actual  1:  21 0 89
```

The earlier row-level split result of 0.8524 accuracy is retained only as a
clearly labelled historical comparison and is not the final performance
claim.

## Architecture

```text
User
  ↓
Flask Web Interface
  ↓
Routes
  ↓
Services
  ↓
Feature Validation / URL Security
  ↓
Associative Classification Model
  ↓
Prediction + Rule Explanation
  ↓
History Database
  ↓
Dashboard / Reports
```

Live URL analysis follows an additional controlled path: the URL is validated,
DNS-resolved, fetched with bounded and revalidated HTTP, and parsed for safe
raw observations. It stops at `mapping_unverified` unless an explicitly
verified provider supplies a complete valid vector. It never substitutes
arbitrary defaults.

## Explainability and dashboard

Feature-vector predictions expose learned association-rule evidence including
antecedent, consequent, support, rule confidence, lift, and matching-rule
counts. Rule confidence is evidence about the training data, not prediction
probability. The evidence is not causal and does not add undocumented
Dataset 379 feature semantics.

The read-only `/dashboard` reports total analyses, Phishy/Suspicious/
Legitimate counts, mapping-unverified records, classification distribution,
recent activity, rule-evidence usage, encoded feature distributions, offline
evaluation metrics, dataset quality, and model/system status. Classification,
status, and date filters are validated server-side. Empty states are explicit,
and the presentation is print-friendly. Feature analytics use stored encoded
values only.

## Demonstration workflow

1. Install dependencies and start with `python app.py`.
2. Open the home page and explain the problem and scope.
3. Open **Feature Vector Test**.
4. Enter a complete valid Dataset 379 encoded vector.
5. Submit it and show the class result.
6. Show the model-derived association-rule evidence.
7. Open history and then the dashboard.
8. Open system status and health.
9. Demonstrate invalid feature input handling.
10. Demonstrate live URL `feature_mapping_unavailable` behavior using a
    permitted test URL; do not claim a fabricated live phishing result.
11. Explain SSRF protections, integrity checks, and the live-mapping
    limitation.

## Final-submission material

The `docs/` directory contains:

- `VIVA_PREPARATION.md`: concise, implementation-grounded viva answers.
- `FINAL_REPORT_OUTLINE.md`: final-year engineering report structure.
- `PRESENTATION_OUTLINE.md`: 15-slide presentation plan.
- `DEMO_EVIDENCE_CHECKLIST.md`: screenshots and evidence to capture manually.

Screenshots and charts are intentionally not fabricated in the repository.
