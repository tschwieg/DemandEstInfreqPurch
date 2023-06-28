# -*- coding: utf-8 -*-

from __future__ import division, print_function
from numpy.random import normal as randn
from sys import version_info
import numpy as np

if version_info >= (3, 0):
    from itertools import reduce

def simData(M, J, ns, alpha, beta, sigma, fac, R, alt = False):
    """Simulate data for MC exercise in Gandhi et al. (2017)

    Args
    ----

    M (int): number of markets
    J (int): number of goods
    ns (int): number of consumer draws
    alpha (float): nuisance parameter in utility equation
    beta (float): parameter of interest in the utility equation
    sigma (float): variance of random coefficients
    fac (float): aux parameter for generating x (var of interest)
    R (int): Number of samples for the random coefficients for x_nonlin

    Returns
    -------

    x (numpy.ndarray): x_{jt}, observed product characteristics
    share (numpy.ndarray): s_{jt}, simulated empirical shares
    mk_id (numpy.ndarray): t, market index
    ns (int): number of consumer draws
    beta (numpy.ndarray): linear parameter vector
    sigma (numpy.ndarray): non-linear parameter diagonal matrix

    Model
    -----

    The model we try to simulate is

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = α_0 + x_jt β_0 + λ_0 x_{jt} v_i + ξ_{jt} + ϵ_{ijt}

    Where
        - α_0, β_0 are alpha and beta
        - x_{jx}, ξ_{jt} are x and xi
        - λ_0, v_i are sigma and rc_seed
        - j, t are pd_id and mk_id

    We do not simulate the model at the individual-product-market level;
    rather, we simulate product market shares and simulate the model at the
    product-market level.
    """

    # Market id
    # ---------

    n     = M * J
    mk_id = np.array([i for i in range(M) for j in range(J)])
    pd_id = np.array([j for i in range(M) for j in range(J)]) + 1

    # Random draws
    # ------------

    if alt:
        p  = (0.99, 0.005, 0.005)
        o  = (1, 12, 15)
        x  = np.random.choice(o, size = (n, 1), p = p)
        xi = (x == 1) * randn(0, 2, (n, 1)) + \
             (x != 1) * randn(0, .1, (n, 1))
    else:
        x  = (pd_id / fac).reshape((n, 1)).astype('float64')
        x += randn(0, 1, (n, 1))       # x_{jt}: Observed product characteristics
        xi = .1 * randn(0, 1, (n, 1))  # ξ_{jt}: Unobserved product quality

    rc_seed = randn(0, 1, (1, R))     # v_i: Random coefficients
    mu_x    = x * sigma * rc_seed     # Individual random utility term
    delta   = alpha + beta * x + xi   # δ_{ij}
    u       = delta + mu_x            # (Simulated) random utility

    # Simulate shares
    # ---------------

    dummies    = np.row_stack((mk_id == t for t in range(M))).astype(int)
    expu       = np.exp(u)                          # Numerator for π_{jt}
    expu_aux   = 1 + dummies.dot(expu)              # Denominator for π_{jt}
    expu_sumj  = np.repeat(expu_aux, [J for i in range(M)], axis = 0)
    share_j    = (expu / expu_sumj).mean(axis = 1)  # π_{jt} prod market shares
    share_0    = 1 - dummies.dot(share_j)           # π_{0t}

    share_true = np.column_stack((share_j.reshape((M, J)), share_0))
    m_gen      = (np.random.multinomial(ns, pi)[:-1] for pi in share_true)
    m_aux      = np.row_stack(m_gen)
    share      = m_aux.flatten().reshape((-1, 1)) / ns

    x_lin  = np.column_stack((np.ones((x.shape[0], 1)), x))
    theta  = np.array([alpha, beta])
    Z      = np.column_stack((x_lin, x ** 2 - 1, x ** 3 - 3 * x))

    return share, x_lin, x, Z, mk_id, ns, sigma, theta, R


def simDataMC(M, J, ns, alpha, beta, sigma, fac, R, alt = False):
    """Simulate data for MC exercise in Gandhi et al. (2017)

    Args
    ----

    M (int): number of markets
    J (int): number of goods
    ns (int): number of consumer draws
    alpha (float): nuisance parameter in utility equation
    beta (float): parameter of interest in the utility equation
    sigma (float): variance of random coefficients
    fac (float): aux parameter for generating x (var of interest)

    Returns
    -------

    x (numpy.ndarray): x_{jt}, observed product characteristics
    xi (numpy.ndarray): ξ_{jt}, unobserved product quality
    rc_seed (numpy.ndarray): v_i, random coefficients
    share (numpy.ndarray): s_{jt}, simulated empirical shares
    mk_id (numpy.ndarray): t, market index
    ns (int): number of consumer draws
    parames (list): list of parameters used for the simulation

    Model
    -----

    The model we try to simulate is

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = α_0 + x_jt (β_0 + λ_0 v_i) + ξ_{jt} + ϵ_{ijt}
                = α_0 + x_jt β_0 + λ_0 x_{jt} v_i + ξ_{jt} + ϵ_{ijt}

    Where
        - α_0, β_0 are alpha and beta
        - x_{jx}, ξ_{jt} are x and xi
        - λ_0, v_i are sigma and rc_seed
        - j, t are pd_id and mk_id

    We do not simulate the model at the individual-product-market level;
    rather, we simulate product market shares and simulate the model at the
    product-market level.
    """

    # Market id
    # ---------

    n      = M * J
    mk_id  = np.array([i for i in range(M) for j in range(J)])
    pd_id  = np.array([j for i in range(M) for j in range(J)]) + 1

    # Random draws
    # ------------

    if alt:
        p  = (0.99, 0.005, 0.005)
        o  = (1, 12, 15)
        x  = np.random.choice(o, size = (n, 1), p = p)
        xi = (x == 1) * randn(0, 2, (n, 1)) + \
             (x != 1) * randn(0, .1, (n, 1))
    else:
        x  = (pd_id / fac).reshape((n, 1)).astype('float64')
        x += randn(0, 1, (n, 1))       # x_{jt}: Observed product characteristics
        xi = .1 * randn(0, 1, (n, 1))  # ξ_{jt}: Unobserved product quality

    rc_seed = randn(0, 1, (1, R))     # v_i: Random coefficients
    mu_x    = x * sigma * rc_seed     # Individual random utility term
    delta   = alpha + beta * x + xi   # δ_{ij}
    u       = delta + mu_x            # (Simulated) random utility

    # Simulate shares
    # ---------------

    dummies    = np.row_stack((mk_id == t for t in range(M))).astype(int)
    expu       = np.exp(u)                          # Numerator for π_{jt}
    expu_aux   = 1 + dummies.dot(expu)              # Denominator for π_{jt}
    expu_sumj  = np.repeat(expu_aux, [J for i in range(M)], axis = 0)
    share_j    = (expu / expu_sumj).mean(axis = 1)  # π_{jt} prod market shares
    share_0    = 1 - dummies.dot(share_j)           # π_{0t}

    share_true = np.column_stack((share_j.reshape((M, J)), share_0))
    m_gen      = (np.random.multinomial(ns, pi)[:-1] for pi in share_true)
    m_aux      = np.row_stack(m_gen)
    share      = m_aux.flatten().reshape((-1, 1)) / ns

    theta = np.array([alpha, beta])
    return x, xi, rc_seed, share, mk_id, ns, sigma, theta


def meanval(exp_delta0, share, x_ind_exp, mk_id, M):
    """Compute the mean utility level

    Args
    ----

    exp_delta0 (numpy.ndarray): exp(δ^0_{.t}) to start the iteration
    share (numpy.ndarray): s_jt product market shares
    x_ind_exp (numpy.ndarray): λ_0 x_{jt} v_i, random coef term
    mk_id (numpy.ndarray): market ID for each product
    M (int): Number of markets

    Returns
    -------
        
    δ^K_{.t}, an estimate of δ_{.t}
    
    Notes
    -----

    This uses the contraction mapping suggested by the BLP, essentially
    iterating the following until convergence

        δ^{k + 1}_{.t} = δ^k_{.t} + log(s_{.t}) - log(s(x_{.t}, δ^k_{.t}))

    We stop at K s.t. ||δ^{K + 1}_{⋅t} - δ^K_{⋅t}|| is below some
    tolerance threshold. For speed, we avoid taking logs and iterate over

        exp(δ^{k + 1}_{.t}) = exp(δ^k_{.t}) * (s_{.t} / s(x_{.t}, δ^k_{.t}))
    """
    select  = [mk_id == t for t in range(M)]
    J       = np.array([sum(mk_id == t) for t in range(M)])
    tol     = 1e-10
    maxK    = 1000
    norm    = 1
    i       = 0
    while (norm > tol) and (i <= maxK):
        # exp(λ_0 x_{jt} v_i + δ_{ij})
        exp_delta = x_ind_exp * exp_delta0

        # s(x_{.t}, δ^k_{.t})
        sx_delta  = np.row_stack((exp_delta[s].sum(0) for s in select))
        share_ind = exp_delta / (1 + np.repeat(sx_delta, J, axis = 0))
        share_mk  = share_ind.mean(axis = 1).reshape((-1, 1))

        # exp(δ^{k + 1}_{.t})
        exp_delta = exp_delta0 * share / share_mk

        # Check convergence
        i    += 1
        norm  = np.abs(exp_delta - exp_delta0).max()
        exp_delta0 = exp_delta

    if i > maxK:
        print("WARNING: Contraction mapping failed; estimates may not be precise.")

    # After convergence, take logs for the actual value
    return np.log(exp_delta)


def mdot(*args):
    """Wrapper to multiply several numpy matrices"""
    return reduce(lambda x, y: np.dot(y, x), args[::-1])
