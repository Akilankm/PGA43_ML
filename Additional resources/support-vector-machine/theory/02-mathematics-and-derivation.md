# Mathematics and Derivation

For labels `y_i in {-1,+1}`, correct classification requires `y_i(w^T x_i+b)>0`.

Point-to-hyperplane distance is `|w^T x+b|/||w||`. Under canonical constraints `y_i(w^T x_i+b)>=1`, margin width is `2/||w||`; maximize margin by minimizing `(1/2)||w||^2`.

## Hard margin
Minimize `(1/2)||w||^2` subject to `y_i(w^T x_i+b)>=1`.

## Soft margin
Introduce `xi_i>=0` and minimize `(1/2)||w||^2 + C sum_i xi_i`, subject to `y_i(w^T x_i+b)>=1-xi_i`. `C` trades margin width against violations.

## Hinge loss
Equivalent unconstrained loss: `max(0,1-y_i(w^T x_i+b))`. For violated margins the subgradient contributes `-y_i x_i` and `-y_i`.

## Dual and support vectors
The dual maximizes `sum alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j x_i^T x_j`, subject to `0<=alpha_i<=C` and `sum alpha_i y_i=0`. Nonzero multipliers identify support vectors.

## Kernel trick
Replace dot products with `K(x,z)=phi(x)^T phi(z)`. Common kernels are linear, polynomial, RBF, and sigmoid. For RBF, large gamma makes local complex boundaries; small gamma makes smooth boundaries. Tune `C` and `gamma` jointly.
