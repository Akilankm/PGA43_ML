# Naive Bayes Masterclass

A complete, student-ready pathway for mastering **Naive Bayes classification** from probability fundamentals to validated implementation.

## Learning outcomes

By the end, a student can:

1. Explain Bayes' theorem, conditional independence, priors, likelihoods, evidence, and posteriors.
2. Derive Gaussian, Multinomial, Bernoulli, and Categorical Naive Bayes.
3. Implement Gaussian and Multinomial Naive Bayes from scratch using numerically stable log-probabilities.
4. Select the appropriate variant for continuous, count, binary, or categorical features.
5. Prevent leakage, diagnose assumption violations, tune smoothing, evaluate calibration, and interpret class evidence.
6. Compare Naive Bayes with logistic regression, linear discriminant analysis, trees, KNN, and SVMs.
7. Reproduce every notebook result with fixed seeds and a documented environment.

## Recommended sequence

1. `theory/01-probability-foundations.md`
2. `theory/02-mathematical-derivation.md`
3. `theory/03-variants-and-data-preparation.md`
4. `theory/04-evaluation-diagnostics-and-tuning.md`
5. `theory/05-failure-modes-comparisons-and-production.md`
6. `src/gaussian_naive_bayes.py`
7. `src/multinomial_naive_bayes.py`
8. `notebooks/naive-bayes-masterclass.ipynb`
9. `exercises/student_exercises.md`
10. `references.md`

## Module map

| Resource | Purpose |
|---|---|
| Theory notes | Concepts, derivations, assumptions, model selection, diagnostics |
| Scratch implementations | Transparent Gaussian and Multinomial estimators |
| Executed notebook | End-to-end experiments with rendered metrics and plots |
| Exercises | Practice, derivations, coding tasks, interview questions |
| Validation script | Confirms notebook execution, outputs, imports, and deterministic tests |

## Quick decision guide

| Data representation | Preferred variant |
|---|---|
| Continuous approximately class-conditionally Gaussian features | GaussianNB |
| Non-negative counts such as word frequencies | MultinomialNB |
| Binary indicators such as word presence | BernoulliNB |
| Integer-coded categorical variables | CategoricalNB |
| Mixed feature families | Separate likelihoods or a different model |

## Core warning

The word *naive* refers to the conditional-independence assumption: features are assumed independent given the class. This is often false, yet classification may still be strong because correct ranking does not require perfectly estimated probabilities.

## Reproducibility

```bash
conda env create -f environment.yml
conda activate naive-bayes-masterclass
python validate_module.py
```

The notebook is committed with executed cells and rendered outputs.