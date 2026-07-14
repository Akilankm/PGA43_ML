# Data Preparation and Mechanics

## Leakage-safe workflow
1. Define prediction unit and target.
2. Create an untouched test split.
3. Fit imputation, encoding, and scaling only inside training folds.
4. Put preprocessing and SVM in one pipeline.
5. Tune with cross-validation.
6. Evaluate once on test.
7. Record seed, data, preprocessing, and package versions.

## Scaling
SVM geometry depends on distances and dot products, so standardization is normally essential.

## Missing and categorical data
Impute inside the pipeline. One-hot encode nominal categories. Linear SVM often works well with sparse high-dimensional matrices.

## Imbalance
Use stratified splits, macro metrics, class-specific recall, class weights, and threshold analysis.

## Linear mechanics
Initialize parameters, compute signed margins, identify violators, update regularization and hinge-loss gradients, iterate, and predict by score sign.

## Kernel mechanics
Scale features, choose a kernel, solve the dual, retain support vectors, and predict through weighted kernel similarities.

## Multiclass
SVC typically uses one-vs-one; linear alternatives may use one-vs-rest.
