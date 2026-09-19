# Demonstration Evidence Checklist

Capture these manually from the running application or command line. Do not
fabricate screenshots or charts.

## Application screens

- [ ] Home page
- [ ] Dataset Feature Vector Test form
- [ ] Valid prediction result
- [ ] Association-rule evidence on the result page
- [ ] Prediction history
- [ ] Security analytics dashboard
- [ ] System status
- [ ] Invalid feature-vector input and error rendering
- [ ] Live analysis with clearly labelled `feature_mapping_unavailable`
- [ ] Health endpoint response

## Project and reproducibility evidence

- [ ] Repository project structure
- [ ] `pytest -q tests` output
- [ ] `git diff --check` output
- [ ] Dataset quality report
- [ ] Reproducibility metadata
- [ ] Model manifest
- [ ] Evaluation report
- [ ] Association-rule artifact
- [ ] Protected artifact SHA-256 verification

## Demonstration safeguards

- [ ] Use a complete valid encoded vector for the prediction demonstration.
- [ ] Label the feature-vector path as testing of an already encoded vector.
- [ ] Do not show a fabricated live phishing classification.
- [ ] Do not display credentials, query strings, or sensitive URL fragments.
- [ ] Explain that rule confidence is not prediction probability.
- [ ] Explain that offline Dataset 379 metrics are not live-web accuracy.

