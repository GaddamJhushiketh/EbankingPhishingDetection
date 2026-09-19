# Final Report Outline

1. Title Page
2. Certificate
3. Declaration
4. Acknowledgement
5. Abstract
6. Table of Contents
7. List of Figures
8. List of Tables
9. Introduction
10. Problem Statement
11. Objectives
12. Existing System
13. Proposed System
14. Literature Survey
15. System Requirements
16. Methodology
17. Dataset
18. Data Preprocessing
19. Associative Classification
20. System Architecture
21. Feature Extraction
22. Security Architecture
23. Implementation
24. Explainability
25. Dashboard
26. Testing
27. Results
28. Limitations
29. Future Scope
30. Conclusion
31. References

## Verified content for the report

Use the UCI Website Phishing Dataset, Dataset 379: 1,353 rows, 10 columns,
nine features, and target `Result`. The target mapping is `-1 = Phishy`,
`0 = Suspicious`, and `1 = Legitimate`. The dataset contains 629 exact
duplicates, 724 unique rows, 48 conflicting feature-only groups, and no
missing values.

Describe the exact feature-plus-target record-group-aware evaluation with
seed 42, 0.20 test target, minimum support 0.01, selected confidence 0.70,
and minimum lift 1.0. Report only the corrected offline metrics:

- Accuracy: `0.7822878228782287`
- Macro precision: `0.8522692601067887`
- Macro recall: `0.6165223665223665`
- Macro F1: `0.6420598680872653`
- Weighted F1: `0.7669808533701776`

The report must distinguish this offline evaluation from live URL analysis.
Live analysis is security-controlled but remains mapping-unavailable when the
complete authoritative Dataset 379 encoding cannot be established.

