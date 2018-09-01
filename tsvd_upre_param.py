#!/usr/bin/env python
import numpy as np
from scipy.optimize import fminbound

def upre_k(sigma, utbn, alpha, k, eta_var):
    '''
    Unbiased Predictive Rist Estimate (UPRE)

    parameters:
    ----------
    sigma: numpy.ndarray, array of full or partial singular values from system matrix
    utbn: numpy.ndarray,  coefficients computed from np.matmul(U.transpose(), b)
    alpha: float, regularization parameter at which to evaluate upre_k
    k: int, number of TSVD components over to evaluate upre_k 
    eta_var: float, estimate of noise variance 

    returns:
    -------
    upre_k: float, UPRE functional evaluated for input parameters
    '''
    gamma = sigma[0:k]**2 / (sigma[0:k]**2 + alpha**2)
    phi   = 1 - gamma
    upre =  2 * eta_var * np.sum(gamma) + np.sum((utbn[0:k] * phi) ** 2)
    return(upre)


def tsvd_upre_parameter(sigma, utbn, eta_var, k_start, k_step, k_max, moving_avg_width = 5, tol = 1e-3, ell = None):
    '''
    Method for estimating optimal index and regularization parameter using
    unbiased predictive risk estimation (UPRE) on truncated singular value decomposition (TSVD)
    by tracking convergence of regularization parameter alpha_k  

    parameters:
    ----------
    sigma: numpy.ndarray, array of full or partial singular values from system matrix
    utbn: numpy.ndarray,  coefficients computed from np.matmul(U.transpose(), b)
    eta_var: float, estimate of noise variance 
    k_start: int, initial value for 
    k_step: int, value by which to increment k
    k_max: int, maximum index to consider
    moving_avg_width: int, width of moving average window for relative changes in alpha_k
    tol: float, convergence tolerance applied to average of relative changes in alpha_k
    ell: int, estimate of index at which noise dominates the Picard coefficients


    returns:
    ----------
    k: int, optimal number of terms in TSVD over which alpha was computed
    alpha_k: float, optimal regularization parameter
    moving_average: float, most recent moving average of relative changers in regularization parameter
    alpha_vector: numpy.ndarray, vector of regularizaton parameters computed 
    '''

    k = k_start
    converged = False
    alpha_vector = np.array([])
    upper_bound = 1.0
    moving_average = np.Inf

    if ell:
        lower_bound = sigma[ell] / np.sqrt(1 - sigma[ell]**2) / 100

    i = 0
    while not converged:

        if not ell:
            lower_bound = sigma[k] / np.sqrt(1 - sigma[k]**2) / 100

        alpha_k = fminbound(lambda alpha: upre_k(sigma, utbn, alpha, k, eta_var), lower_bound,upper_bound)
        alpha_vector = np.append(alpha_vector, alpha_k)
        k += k_step

        if i >= moving_avg_width:
            prev_estimates = alpha_vector[(-moving_avg_width-1):-1]
            succ_estimates = alpha_vector[-moving_avg_width:]
            relative_changes = np.abs(prev_estimates - succ_estimates) / prev_estimates
            moving_average = np.mean(relative_changes)

        if (k >= k_max) or ((moving_average < tol) and (abs(lower_bound - alpha_k) > 1e-3)):
            converged = True

        i += 1

    return(k, alpha_k, moving_average, alpha_vector)