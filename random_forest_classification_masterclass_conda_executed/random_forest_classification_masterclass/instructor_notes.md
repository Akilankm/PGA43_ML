# Instructor Notes — Random Forest Classification Masterclass

## Teaching sequence

1. Start from the decision objective and asymmetric false-negative cost.
2. Derive the algorithm mechanics before using scikit-learn.
3. Audit every feature and establish the synthetic-data boundary.
4. Separate train, validation, and test before preprocessing or model selection.
5. Use the baseline ladder to establish incremental value.
6. Diagnose capacity and variance instead of treating tuning as a black box.
7. Distinguish ranking quality, probability quality, and threshold policy.
8. Interpret importance as model reliance, never causality.
9. Reserve the test set for one final evaluation.
10. End with serialization, limitations, and monitoring questions.

## Classroom checkpoints

- Calculate one impurity or bootstrap example by hand.
- Explain why validation and test have different roles.
- Compare ROC-AUC with average precision under imbalance.
- Show how the threshold changes when the cost ratio changes.
- Treat subgroup tables as diagnostics, not fairness certification.

## Common misconceptions

- A visible split is not a causal mechanism.
- High training accuracy is not evidence of generalization.
- Feature importance does not identify a treatment effect.
- OOB performance does not replace a final untouched test.
- A threshold of 0.5 is not automatically operationally correct.
