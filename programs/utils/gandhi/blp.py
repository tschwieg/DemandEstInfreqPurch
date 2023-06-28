# -*- coding: utf-8 -*-

from __future__ import division, print_function
from numpy.random import normal as randn
from scipy.optimize import fmin
from numpy.linalg import solve
import numpy as np

from .common import mdot, meanval

def estimateBLP(share, x_lin, x_nonlin, z, mk_id, ns, R, alt = False):
    """Estimate BLP model

    Args
    ----

    share (numpy.ndarray): s_{jt}, prod market shares
    x_lin (numpy.ndarray): x^1_{jt}, prod chars that enter utility linearly
    x_nonlin (numpy.ndarray): x^2_{jt}, prod chars that enter utility non-linearly
    z (numpy.ndarray): z_{jt}, instruments for x_lin
    mk_id (numpy.ndarray): Market id
    ns (numpy.ndarray): Number of goods per market
    R (int): Number of samples for the random coefficients for x_nonlin

    Returns
    -------

    This function returns estimates for β_0, Σ_0 in the model

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = x_jt β_0 + x_{jt} Σ_0 v_i + ξ_{jt} + ϵ_{ijt}

    using the methods proposed by Berry, Levinsohn, and Pakes (1995) and
    following the guide by Nevo (2000). β_0 is the vector of parameters that
    will enter the objective linearly, while Λ_0 is the diagonal matrix with
    parameters that will enter the objective non-linearly.  This is currently
    a work in progress; the features of this function are a subset of the
    features that should be available when estimating this model.

    References
    ----------

    - Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices In
      Market Equilibrium. Econometrica, 63(4), 841.

    - Nevo, A. (2000). A Practitioner’s Guide to Estimation of
      Random-Coefficients Logit Models of Demand. Journal of Economics &
      Management Strategy, 9(4), 513–548.
    """

    # Selected data
    # -------------

    sele_ind      = (share.flatten() > 0)
    x_lin_sele    = x_lin[sele_ind]
    x_nonlin_sele = x_nonlin[sele_ind]
    share_sele    = share[sele_ind]
    mk_id_sele    = mk_id[sele_ind]
    z_sele        = z[sele_ind]

    # IV helper
    # ---------

    invZZ  = np.linalg.pinv(z_sele.T.dot(z_sele))
    xz     = x_lin_sele.T.dot(z_sele)
    invXZZ = np.linalg.pinv(mdot(xz, invZZ, xz.T))
    invXZ  = mdot(invXZZ, xz.dot(invZZ))

    # Starting value of ξ_{jt}
    # ------------------------

    M        = len(np.unique(mk_id))
    select   = [mk_id_sele == t for t in range(M)]
    J        = np.array([sum(mk_id_sele == t) for t in range(M)])
    sx_delta = np.row_stack((share_sele[s].sum(0) for s in select))
    share0   = 1 - np.repeat(sx_delta, J, axis = 0)
    y_sele   = np.log(share_sele / share0)

    beta0      = mdot(invXZ, z_sele.T.dot(y_sele))
    delta_sele = x_lin_sele.dot(beta0)
    exp_delta  = np.exp(delta_sele)

    # v_i: Simulate Random coefficients
    # ---------------------------------

    n, k1 = x_lin_sele.shape
    n, k2 = x_nonlin_sele.shape
    v     = randn(0, 1, (k2, R))

    # Actual optimization
    # -------------------

    global beta
    sigma   = np.array([1 for i in range(k2)])
    beta    = np.array([0 for i in range(k1)])
    argsBLP = (share_sele,
               x_lin_sele,
               x_nonlin_sele,
               exp_delta,
               v,
               z_sele,
               invZZ,
               invXZ,
               mk_id_sele,
               M)

    if alt:
        objectiveBLP(np.array([0.5]), *argsBLP)
        sigma_blp = np.array([0.5])
    else:
        kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}
        sigma_blp = fmin(func = objectiveBLP,
                         args = argsBLP,
                         x0   = sigma,
                         **kwargsOptim)

    return sigma_blp, beta


def objectiveBLP(sigma, share, x_lin, x_nonlin, exp_delta, v, z, invZZ, invXZ, mk_id, M):
    """BLP Objective Function

    Args
    ----

    sigma (numpy.ndarray): diagnonal elements of Σ, the standard deviation of rand coefs
    share (numpy.ndarray): empirical shares
    x_lin (numpy.ndarray): x_{jt} observables that enter the objective linearly
    x_nonlin (numpy.ndarray): x_{jt} observables that enter the objective non-linearly
    exp_delta (numpy.ndarray): exp(δ_{jt})
    v (numpy.ndarray): v_i, random coefficients
    z (numpy.ndarray): Z, Instrument
    invZZ (numpy.ndarray): auxiliary array; (Z' Z)^-1 weight matrix
    invXZ (numpy.ndarray): auxiliary array; [X' Z (Z' Z)^-1 Z' X]^-1 X' Z (Z' Z)^-1
    mk_id (numpy.ndarray): market ID
    M (int): number of markets

    Returns
    -------

    BLP objective, namely

        J(θ) = ω(θ)' Z (Z' Z)^-1 Z' ω(θ)

    with ω(θ) = (δ_{jt} - x_{jt} θ)

    Notes
    -----

    The function computes the objective for the BLP model. Ultimately we aim
    to estimate

        θ = argmin J(θ)

    First we obtain an estimate for δ_{jt} using an iterative procedure (see
    meanval for more on that point), and then we solve for θ in

        δ_{jt} = x_{jt} θ + ξ_{jt}
        δ_{ij} = α_0 + x_jt β_0 + ξ_{jt}

    Normally we could obtain a solution via the system of equations

        Δ = X θ

    However, endogeneity is a chief concern in this model. The point of
    BLP, as I understand it, is to transform the problem to allow us to use
    instruments in a setting where market shares and product characteristics
    are the primary variables available. In this case, the IV approach
    involves finding z such that

        E[z ω(θ*)] = 0
        ω(θ) = (δ_{jt} - x_{jt} θ)

    with θ* the true value. Hence

        E[z (δ_{jt} - x_{jt} θ*)] = 0

    and the solution to the system

        Z' Δ = Z' X θ

    will be consistent. Note that by assumption we have

        E[z' ξ_{jt} ξ_{jt} z] = E[z' E[ξ_{jt} ξ_{jt} | z] z]
                              = E[z' z] σ^2_ξ

    Hence we can obtain our estimate by solving

        C(θ) = (Z' Δ - Z' X θ)
        θhat = argmin C(θ)' (Z' Z)^-1 C(θ)

    The solution to this is known and given by

        θhat = [X' Z (Z' Z)^-1 Z' X]^-1 [X' Z (Z' Z)^-1 Z' Δ]

    (My understanding is that in this case, the variance of ξ_{jt} is not
    identified but it does not affect the estimation of θ.) Finally, we can
    estimate the unobserved effect ξ_{jt} as the residuals of this model with
    Z' (Δ - X θhat). Now that we have all the moving parts, we can finally
    compute the objective, that is,

        J(θhat) = ω(θhat)' Z (Z' Z)^-1 Z' ω(θhat)

    with ω(θhat) = (δ_{jt} - x_{jt} θhat) as above and δ_{jt} estimated
    from meanval.
    """
    global beta

    # exp(x_{jt} Σ_0 v_i)
    x_ind_exp = np.exp((sigma[:, None] * v) * x_nonlin)

    # Estimate δ_{jt} using iterative prodedure
    delta = meanval(exp_delta, share, x_ind_exp, mk_id, M)

    # Obtain GLS estimate of θhat
    deltaz = z.T.dot(delta)
    beta   = mdot(invXZ, deltaz)

    # Compute ξ_{jt} as the residuals from the model
    xi  = delta - x_lin.dot(beta)
    xiz = xi.T.dot(z)

    # Return the objective J(θ)
    return float(mdot(xiz, invZZ, xiz.T))


def estimateBLPMC(x, xi, rc_seed, share, mk_id, ns, sigma, beta0, alt = False):
    """Estimate BLP model; see estimateBLP for details."""

    # Setup BLP estimation
    # --------------------

    # zero_frac = np.mean(share == 0)
    sele_ind  = (share.flatten() > 0)

    # Selected data
    # -------------

    x_sele      = x[sele_ind]
    share_sele  = share[sele_ind]
    x1_sele     = np.column_stack((np.ones((sele_ind.sum(), 1)), x_sele))
    mk_id_sele  = mk_id[sele_ind]

    # Construct IV
    # ------------

    IV_gen   = (x1_sele, x_sele ** 2 - 1, x_sele ** 3 - 3 * x_sele)
    IV_blp   = np.column_stack(IV_gen)
    invA_blp = np.linalg.pinv(IV_blp.T.dot(IV_blp))

    # Starting value of ξ_{jt}
    # ------------------------

    xi_sele       = xi[sele_ind]
    delta_sele    = x1_sele.dot(beta0).reshape((-1, 1)) + xi_sele
    exp_delta_blp = np.exp(delta_sele)

    # Actual optimization
    # -------------------

    M = len(np.unique(mk_id))
    argsBLP = (x_sele,
               x1_sele,
               exp_delta_blp,
               rc_seed,
               IV_blp,
               invA_blp,
               share_sele,
               mk_id_sele,
               M)

    global beta
    beta = np.empty(beta.shape)

    if alt:
        objectiveBLPMC(sigma, *argsBLP)
        sigma_blp = sigma
    else:
        kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}
        sigma_blp = fmin(func = objectiveBLPMC,
                         args = argsBLP,
                         x0   = sigma + 0.1,
                         **kwargsOptim)

    return sigma_blp, beta


def objectiveBLPMC(sigma, x, x1, exp_delta, rc_seed, IV, invA, share, mk_id, M):
    """BLP Objective Function; see objectiveBLP for details.

    Args
    ----

    sigma (float): λ, variance on random coefficients
    x (numpy.ndarray): x_{jt} observables
    x1 (numpy.ndarray): auxiliary array; x_{jt} with a constant
    exp_delta (numpy.ndarray): exp(δ_{jt})
    rc_seed (numpy.ndarray): v_i, random coefficients
    IV (numpy.ndarray): Z, Instrument
    invA (numpy.ndarray): auxiliary array; (Z' Z)^-1 weight matrix
    share (numpy.ndarray): empirical shares
    mk_id (numpy.ndarray): market ID
    M (int): number of markets

    """
    global beta

    # exp(λ_0 x_{jt} v_i)
    rc_sd     = sigma * rc_seed
    mu_x      = rc_sd * x
    x_ind_exp = np.exp(mu_x)

    # Estimate δ_{jt} using iterative prodedure
    delta = meanval(exp_delta, share, x_ind_exp, mk_id, M)

    # Obtain GLS estimate of θhat
    temp1 = x1.T.dot(IV)
    temp2 = delta.T.dot(IV)
    beta  = solve(mdot(temp1, invA, temp1.T), mdot(temp1, invA, temp2.T))

    # Compute ξ_{jt} as the residuals from the model
    resid = delta - x1.dot(beta)
    temp  = resid.T.dot(IV)

    # Return the objective J(θ)
    return float(mdot(temp, invA, temp.T))
