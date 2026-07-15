# Mathematics and Derivation

For labels y_i in {-1,+1}, AdaBoost builds F_M(x)=sum_m alpha_m h_m(x) and predicts sign(F_M(x)). At round m, fit h_m using sample weights w_i. Weighted error is epsilon_m=sum_i w_i I(y_i != h_m(x_i)). Learner weight is alpha_m=0.5 ln((1-epsilon_m)/epsilon_m). Update w_i <- w_i exp(-alpha_m y_i h_m(x_i)) and normalize.

Correct observations are downweighted; mistakes are upweighted. AdaBoost minimizes exponential loss sum_i exp(-y_i F(x_i)). The signed quantity y_i F(x_i) is the margin. A binary weak learner must achieve epsilon_m < 0.5. Errors are clipped numerically when they approach zero or one.
