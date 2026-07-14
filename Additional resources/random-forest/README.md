# Random Forest Masterclass

Random Forest is an ensemble learning method that combines many decorrelated decision trees through bootstrap aggregation and random feature selection. This module develops the method from first principles, implements an educational classifier with NumPy, validates it against scikit-learn, and teaches responsible model selection and diagnosis.

## Learning outcomes
Students will be able to explain bagging, bootstrap samples, random subspaces, impurity reduction, majority voting, out-of-bag evaluation, feature importance, calibration, bias-variance trade-offs, and production constraints; implement a forest from scratch; build and tune leakage-safe library pipelines; and decide when Random Forest is or is not suitable.

## Learning sequence
1. `theory/01-foundations.md`
2. `theory/02-mathematics-and-derivation.md`
3. `theory/03-training-and-data-preparation.md`
4. `theory/04-diagnostics-and-evaluation.md`
5. `theory/05-tuning-comparisons-and-production.md`
6. `src/random_forest_from_scratch.py`
7. `notebooks/random-forest-masterclass.ipynb`
8. `exercises/student_exercises.md`
9. `references.md`

## Reproduce
```bash
conda env create -f environment.yaml
conda activate random-forest-masterclass
python build_executed_notebook.py
python validate_module.py
```
