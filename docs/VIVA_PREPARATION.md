# Viva Preparation

These answers describe the implementation in this repository. They do not
claim undocumented Dataset 379 feature semantics or live-web accuracy.

## Questions and answers

1. **What is phishing?**  
   Phishing is a social-engineering attack in which a fraudulent site or
   message imitates a trusted service to obtain information such as banking
   credentials.

2. **Why is e-banking phishing dangerous?**  
   It can expose credentials, account data, and financial transactions.

3. **What is associative classification?**  
   It combines association-rule mining with class prediction: feature-value
   combinations are mined and rules with class consequents are used to
   classify records.

4. **Why use association rules?**  
   Rules expose the feature combinations supporting a classification, making
   the model easier to inspect than an opaque score.

5. **What is Apriori?**  
   Apriori finds frequent itemsets and uses them to generate association
   rules subject to support and confidence thresholds.

6. **What are support, confidence, and lift?**  
   Support is transaction frequency, confidence is the observed frequency of
   the consequent given the antecedent, and lift compares that frequency with
   the consequent baseline frequency.

7. **What dataset was used?**  
   The UCI Website Phishing Dataset, Dataset 379, with 1,353 rows and 10
   columns.

8. **Why Dataset 379?**  
   It is the project’s authoritative, documented source for the nine encoded
   website features and three-class `Result` target.

9. **What are the nine features?**  
   `SFH`, `popUpWidnow`, `SSLfinal_State`, `Request_URL`,
   `URL_of_Anchor`, `web_traffic`, `URL_Length`, `age_of_domain`, and
   `having_IP_Address`.

10. **What is the target mapping?**  
    `-1 = Phishy`, `0 = Suspicious`, `1 = Legitimate`.

11. **Why are there duplicate records?**  
    The raw dataset contains 629 exact duplicates. They are preserved for the
    primary rule-support workflow and separately analyzed after deduplication.

12. **Why use group-aware evaluation?**  
    Exact feature-plus-target records must not appear in both train and test,
    otherwise memorization can inflate the result.

13. **What is data leakage?**  
    Leakage occurs when information from evaluation records or their exact
    duplicates influences training, producing an over-optimistic metric.

14. **What is the final model accuracy?**  
    The corrected offline Dataset 379 accuracy is
    `0.7822878228782287`.

15. **Why should 85.24% not be reported as the final result?**  
    It came from a historical row-level split affected by duplicate overlap.
    The project reports the corrected exact-record-group evaluation instead.

16. **What is macro F1?**  
    It is the unweighted mean of the F1 score for each class, so each class
    contributes equally regardless of support.

17. **Why is Suspicious recall relatively low?**  
    The Suspicious class has only 103 source records and 21 test records, and
    the learned rules identify that class less completely than the other
    classes. This is a measured limitation, not a fabricated explanation.

18. **What is explainability here?**  
    The system shows matching learned rules with antecedent, consequent,
    support, confidence, lift, and deterministic ranking.

19. **What is SSRF?**  
    Server-side request forgery tricks a server into requesting an unintended
    internal or protected destination.

20. **How does the system prevent SSRF?**  
    It restricts schemes, rejects credentials and internal hostnames, resolves
    DNS, blocks private/loopback/link-local/reserved/multicast/unspecified
    addresses, validates every redirect, and bounds requests.

21. **Why is URL redaction needed?**  
    History and result views should not retain credentials, query strings, or
    fragments that may contain sensitive values.

22. **What happens when live mappings are unavailable?**  
    The application stores safe raw observations as
    `feature_mapping_unavailable` and does not invoke the classifier.

23. **Why not fabricate missing feature values?**  
    A guessed encoding would make the prediction scientifically unsupported
    and could mislead users.

24. **How is model integrity protected?**  
    The application hashes the application-controlled pickle before loading
    it, compares the hash with `hmac.compare_digest`, and requires
    `MODEL_SHA256` in production.

25. **Why use Flask?**  
    Flask provides a small, modular Python web layer that fits the project’s
    routes, templates, services, and testable application factory.

26. **Why use MySQL/SQLite compatibility?**  
    SQLite gives deterministic local development and testing; the configured
    SQLAlchemy URI also supports MySQL deployment.

27. **What is the role of mlxtend?**  
    It supplies Apriori frequent-itemset mining and association-rule
    generation used by the classifier.

28. **What are the limitations?**  
    The live nine-feature operational mappings are not fully verified;
    Suspicious recall is relatively low; the model is an offline Dataset 379
    classifier, not a claim of current live-web accuracy; and the pickle is a
    trusted application artifact.

29. **What are future enhancements?**  
    A primary-evidence-backed feature provider could be integrated only after
    all nine mappings are scientifically verified. Further work could also
    expand data, calibrate evaluation, and add deployment observability.

30. **What is the difference between offline evaluation and live detection?**  
    Offline evaluation classifies already encoded Dataset 379 records under a
    fixed split. Live analysis safely observes a URL but currently cannot
    establish the complete authoritative encoding, so it stops without a
    prediction.

