# Presentation Outline

Suggested 15-slide final presentation:

1. **Title** — project title, student, guide, institution.
2. **Problem** — e-banking phishing risk and the detection challenge.
3. **Motivation** — need for transparent and reproducible classification.
4. **Objectives** — dataset validation, associative rules, security, and UI.
5. **Existing vs Proposed System** — limitations of opaque or unsafe flows
   versus this controlled workflow.
6. **System Architecture** — Flask, routes, services, model, database, and
   dashboard.
7. **Dataset** — UCI Dataset 379, dimensions, nine features, target mapping,
   duplicates, and quality findings.
8. **Associative Classification Methodology** — preprocessing, Apriori,
   support, confidence, lift, filtering, matching, and fallback.
9. **Security Architecture** — SSRF controls, bounded HTTP, redaction, and
   integrity verification.
10. **Implementation / UI** — home, feature-vector test, result, history, and
    status screens.
11. **Explainability** — model-derived matching rule evidence and its limits.
12. **Dashboard** — counts, activity, rules, encoded values, metadata, and
    safe filters.
13. **Results** — corrected offline evaluation only:
    accuracy 0.7823, macro F1 0.6421, weighted F1 0.7670.
14. **Limitations and Future Scope** — unresolved live mappings, class
    imbalance, offline/live distinction, and evidence-backed future work.
15. **Conclusion / Q&A** — contributions, safeguards, and questions.

Do not present fabricated screenshots, charts, probabilities, or live
phishing predictions. Use screenshots captured from the running application.

