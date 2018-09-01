## TSVD UPRE Parameter Estimation
___

This repository contains companion code for the paper _Convergence of Regularization Parameters for Solutions using the Filtered Truncated Singular Value Decomposition_ by Rosemary A. Renaut, Anthony Helmstetter, and Saeed Vatankhah.

In solving the linear inverse problem $A x = b_\text{true} + \eta = b$ with $\eta \sim  \mathcal{N} (0, \sigma^2 I)$ via Tikhonov regularization of the form 
$$ x^* = \underset{x}{\operatorname{argmin}} \{ \|A x - b \|^2 + \alpha^2 \| x \|\} $$

this module may be used to provide an estimate $\alpha_k$ to the optimal $\alpha^*$ obtained by minimization of the Unbiased Predictive Risk Estimation (UPRE) functional. As opposed to explicitly computing $\alpha^*$ by minimizing the UPRE functional over the full SVD, this module will determine $\alpha_k \approx a^*$ by minimizing the UPRE functional over the TSVD of $A$ of size $k$ for increasing $k$, tracking the convergence properties of $\alpha_k$ as it does so, returning $k$ and $\alpha_k$ upon convergence. 

 The python source ```tsvd_upre_param.py``` contains two functions  ```upre_trunc``` and ```tsvd_upre_parameter```.

The function ```tsvd_trunc``` takes as input:
- ```sigma```, the singular values of $A$
- ```utbn```, the Picard coefficients $U^T b$
-  ```alpha```, scalar value $\alpha$
-  ```k```, index $k$
- ```eta_var```, an estimate of the noise variance $\sigma^2$
  
and returns the UPRE function value for the given input. The function ```tsvd_upre_parameter``` takes as input:
- ```sigma```, the singular values of $A$
- ```utbn```, the Picard coefficients $U^T b$
- ```eta_var```, an estimate of the noise variance $\sigma^2$
- ```k_start```, a starting indexing for $k$
- ```k_step```, a value by which to increment $k$
- ```k_max```, a maximum allowable value for $k$
- ```moving_avg_width```, the width of the moving window over which to compare relative changes in $\alpha_k$
- ```tol``` tolerance for convergence
- (optional) ```ell``` to use in determining a lower bound over which to search for $\alpha_k$

and returns the optimal $k$ and $\alpha_k$ from which the solution $x$ can be computed. 

A paper manuscript will soon be provided for algorithm details.