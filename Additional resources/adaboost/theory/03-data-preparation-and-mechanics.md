# Data Preparation and Mechanics

Create an untouched test split. Keep preprocessing inside training folds. Initialize equal observation importance, fit a weak learner, measure its weighted error, compute its contribution, update observation importance, normalize, and repeat. Aggregate the learners through weighted voting.

Tree stumps normally do not require scaling. Missing values require imputation unless the base learner supports them. Encode categorical variables inside the modeling pipeline. For imbalanced targets, use stratified splits and balanced metrics, and distinguish class-cost adjustments from AdaBoost's adaptive emphasis on mistakes.
