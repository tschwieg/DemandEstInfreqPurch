# -*- coding: utf-8 -*-

from __future__ import division, print_function

# from utils.randcoef.squarem import squarem, fixed_point
from .squarem import squarem, fixed_point

from numpy.random import normal as randn
from functools import partial
from sys import version_info
import pandas as pd
import numpy as np

if version_info >= (3, 0):
    from functools import reduce


def meanval(exp_delta0, shareLap, exp_x, mkID, M,
            contractionMethod = "squarem", contractionIteration = 3,
            **kwargs):
    """Compute the mean utility level via fixed point iteration

    Args
    ----

    exp_delta0 : numpy.ndarray
        exp(δ^0_{.t}) to start the iteration
    shareLap : numpy.ndarray
        s_jt product market shares
    exp_x : numpy.ndarray
        λ_0 x_{jt} v_i, random coef term
    mkID : numpy.ndarray
        market ID for each product
    M : int
        Number of markets
    contractionMethod : str
        - "squarem": Use the SQUAREM to speed up the δ_{jt} contraction mapping
                     (Varadhan and Roland, 2008)
        - "iteration": Use a regular fixed-point iteration to compute the
                       δ_{jt} contraction mapping
    contractionIteration : int
        How to compute the mean utility. Different methods are equivalent but
        have different speeds.

    kwargs are passed to the iteration routine.

    Returns
    -------

    δ^K_{jt}, an estimate of δ_{jt}

    Notes
    -----

    This uses the contraction mapping suggested by the BLP, essentially
    iterating the following until convergence

        δ^{k + 1}_{.t} = δ^k_{.t} + log(s_{.t}) - log(s(x_{.t}, δ^k_{.t}))

    We stop at K s.t. ||δ^{K + 1}_{⋅t} - δ^K_{⋅t}|| is below some
    tolerance threshold. For speed, we avoid taking logs and iterate over

        exp(δ^{k + 1}_{.t}) = exp(δ^k_{.t}) * (s_{.t} / s(x_{.t}, δ^k_{.t}))

    Since this iterations amounts to finding a fixed point of the above
    mapping, we can use acceleration methods available for fixed-point
    iterations. In particular we default to the SQUAREM method proposed by
    Varadhan and Roland (2008).

    References
    ----------

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.
    """
    n, k   = exp_x.shape
    index  = np.arange(n)
    selint = [index[mkID == t] for t in range(M)]
    kwargsMeanval = {"lnorm": meanvalSupNorm, "xtol": 1e-14}
    kwargsMeanval.update(kwargs)

    if contractionIteration == 3:
        meanvalFun = meanvalMapping3
    elif contractionIteration == 2:
        meanvalFun = meanvalMapping2
    else:
        meanvalFun = meanvalMapping

    if contractionMethod == "squarem":
        methodFun = squarem
    else:
        methodFun = fixed_point

    results = methodFun(meanvalFun,
                        exp_delta0,
                        args = (shareLap, exp_x, selint),
                        **kwargsMeanval)

    # After convergence, take logs for the actual value
    if not results.success:
        print("WARNING: Contraction mapping failed. Results may not be accurate.")

    return np.log(results.x)


def meanvalMapping2(exp_delta0, shareLap, exp_x, selint):
    return np.row_stack(map(meanvalPool2, [(exp_delta0[s], shareLap[s], exp_x[s]) for s in selint]))


def meanvalPool2(args):
    return meanvalPoolAux(*args)


def meanvalMapping3(exp_delta0, shareLap, exp_x, selint):
    return np.row_stack(map(partial(meanvalPool3, exp_delta0 = exp_delta0, shareLap = shareLap, exp_x = exp_x), selint))


def meanvalPool3(s, exp_delta0, shareLap, exp_x):
    return meanvalPoolAux(exp_delta0[s], shareLap[s], exp_x[s])


def meanvalPoolAux(exp_delta0, shareLap, exp_x):
    exp_delta1 = exp_x * exp_delta0
    shareInd   = exp_delta1 / (1 + exp_delta1.sum(0))
    shareMkt   = shareInd.mean(axis = 1).reshape((-1, 1))
    return exp_delta0 * shareLap / shareMkt


def meanvalMapping(exp_delta0, shareLap, exp_x, selint):
    """Mapping for computing mean utility

    The function computes the mapping

        δ -> δ + log(s_{jt}) - log(s(x_{jt}, δ^k_{jt}))

    where s_{jt} are the empirical shares and s(.) are the simulated product
    market shares.

    Args
    ----

    exp_delta0 : numpy.ndarray
        exp(δ^0_{jt}) to start the iteration
    shareLap : numpy.ndarray
        s_jt product market shares
    exp_x : numpy.ndarray
        λ_0 x_{jt} v_i, random coef term
    mkID : numpy.ndarray
        market ID for each product
    selint : iterable
        set with indexes for each market
    """

    # exp(λ_0 x_{jt} v_i + δ_{ij})
    exp_delta1 = exp_x * exp_delta0
    # exp_delta1 = exp_x * np.exp(delta0)

    # s(x_{.t}, δ^k_{.t})
    # sx_gen    = (1 / (1 + exp_delta1[s].sum(0)) for s in selint)
    # sx_delta1 = np.row_stack(sx_gen)
    # shareInd  = exp_delta1 * sx_delta1[mkID]
    # shareMkt  = shareInd.mean(axis = 1).reshape((-1, 1))
    shareInd  = np.row_stack((exp_delta1[s] / (1 + exp_delta1[s].sum(0)) for s in selint))
    shareMkt  = shareInd.mean(axis = 1).reshape((-1, 1))

    # exp(δ^{k + 1}_{.t})
    return exp_delta0 * shareLap / shareMkt


def meanvalSupNorm(x, y):
    """Sup norm for exp(δ^{k + 1}_{jt}), exp(δ^k_{jt}) to return ||δ^{k + 1}_{jt} - δ^k_{jt}||"""
    return np.abs(np.log(x / y)).max()


def meanvalContinue(x, y):
    """Dummy function to iterate until the maximum number of iterations is reached."""
    return 1


def mdot(*args):
    """Wrapper to multiply several numpy matrices"""
    return reduce(lambda x, y: np.dot(y, x), args[::-1])


def getDummies(df, cols = None):
    """Dummies from multiple columns with arbitrary data

    WARNING: This returns a numpy matrix treating each row as a single
    category (vs treating each column independently).

    Args
    ----

    df : pd.DataFrame
        Data frame with the variables to convert

    Kwargs
    ------

    cols : list of strings
        Columns to turn into dummies (all by default)

    Returns
    -------

    numpy matrix with dummies
    """
    return pd.get_dummies(encodeVariables(df, cols)).values


def encodeVariables(df, cols = None):
    """Wrapper for pandas.factorize to encode multiple variables

    Args
    ----

    df : pd.DataFrame
        Data frame with the variables to convert

    Kwargs
    ------

    cols : list of strings
        Columns to turn into dummies (all by default)

    Returns
    -------

    numpy array with encoding
    """
    if cols is None:
        cols = df.columns

    K = len(cols)
    tmpEncoded = np.column_stack((pd.factorize(df[col])[0] for col in cols))
    tmpRanges  = tmpEncoded.max(0) + 1
    bijection  = tmpEncoded[:, 0]
    for k in range(1, K):
        bijection = tmpRanges[k] * bijection + tmpEncoded[:, k]

    return bijection


def simData(M, J, ns, alpha, beta, sigma, pi, R, alt = False, price = 0):
    """Simulate data for random coefficient logit model

    Args
    ----

    M : int
        number of markets
    J : int
        number of goods
    ns : int
        number of consumer draws
    beta : numpy.ndarray
        linear parameters
    sigma : numpy.ndarray
        variance of random coefficients
    pi : numpy.ndarray
        Transition matrix for consumer characteristics
    R : int
        Number of samples for the random coefficients for x_nonlin;
        this is also used for the consumer demographics.

    Returns
    -------

    share : numpy.ndarray
        s_{jt}, simulated empirical shares
    xLin : numpy.ndarray
        x_{jt}, observed product characteristics that enter lienarly
    xNonLin : numpy.ndarray
        x_{jt}, observed product characteristics that enter non-linearly
    Z : numpy.ndarray
        z_{jt}, instruments
    dCons : numpy.ndarray
        D_i, consumer demographics
    mkID : numpy.ndarray
        t, market ID
    ns : int
        number of consumer draws
    beta : numpy.ndarray
        linear parameter vector
    sigma : numpy.ndarray
        non-linear parameter diagonal matrix
    pi : numpy.ndarray
        non-linear parameter transition matrix
    R : int
        Number of consumer draws

    Model
    -----

    The model we try to simulate is

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = x_jt β_0 + x_{jt} Σ_0 v_i + x_{jt} Π_0 D_i + ξ_{jt} + ϵ_{ijt}

    Where
        - β_0 are the linear parameters
        - Σ_0 is the standard deviation of the random coefficients
        - Π_0 is the standard deviation of the random coefficients
        - x_{jx} are the product characteristics
        - ξ_{jt} are the product-market level errors
        - v_i are the random coefficients
        - D_i are consumer characteristics
        - j, t denote a product-market

    We do not simulate the model at the individual-product-market level;
    rather, we simulate product market shares and simulate the model at the
    product-market level.
    """

    # Market id
    # ---------

    n    = M * J
    mkID = np.array([i for i in range(M) for j in range(J)])
    pdID = np.array([j for i in range(M) for j in range(J)]) + 1
    K    = len(sigma)
    K    = len(beta)

    if pi is not None:
        K, d = pi.shape

    # Random draws
    # ------------

    if alt:
        p  = (0.99, 0.005, 0.005)
        o  = (1., 12., 15.)
        x  = np.random.choice(o, size = (n, K), p = p)
        xi = (x == 1) * randn(0, 2, (n, 1)) + \
             (x != 1) * randn(0, .1, (n, 1))
    else:
        x  = (pdID / 10).reshape((n, 1)).astype('float64')
        x  = x + randn(0, 1, (n, K))   # x_{jt}: Observed product characteristics
        xi = .1 * randn(0, 1, (n, 1))  # ξ_{jt}: Unobserved product quality

    xNonLin = x
    if price != 0:
        z    = randn(0, 1, (n, 1))
        pr   = z + xi + 0.1 * randn(0, 1, (n, 1))
        x    = np.column_stack((pr, x))
        beta = np.concatenate(([price], beta))

    vi      = randn(0, 1, (K, R))               # v_i: Random coefficients
    mu_x    = xNonLin.dot(sigma[:, None] * vi)  # Individual random utility term
    if pi is not None:
        Di      = randn(0, 1, (d, R))           # D_i: Consumer demographics
        mu_x   += xNonLin.dot(pi.dot(Di))
    else:
        Di      = None

    delta   = x.dot(beta).reshape((-1, 1))  # δ_{ij}
    delta  += alpha + xi                    # Ibid.
    u       = delta + mu_x                  # (Simulated) random utility

    # Simulate shares
    # ---------------

    dummies    = np.row_stack((mkID == t for t in range(M))).astype(int)
    expu       = np.exp(u)                          # Numerator for π_{jt}
    expu_aux   = 1 + dummies.dot(expu)              # Denominator for π_{jt}
    expu_sumj  = np.repeat(expu_aux, [J for i in range(M)], axis = 0)
    share_j    = (expu / expu_sumj).mean(axis = 1)  # π_{jt} prod market shares
    share_0    = 1 - dummies.dot(share_j)           # π_{0t}

    share_true = np.column_stack((share_j.reshape((M, J)), share_0))
    m_gen      = (np.random.multinomial(ns, sj)[:-1] for sj in share_true)
    m_aux      = np.row_stack(m_gen)
    share      = m_aux.flatten().reshape((-1, 1)) / ns

    xLin   = np.column_stack((np.ones((n, 1)), x))
    theta  = np.concatenate(([alpha], beta))
    if price != 0:
        z = np.column_stack((z, x, xNonLin ** 2 - 1, xNonLin ** 3 - 3 * xNonLin))
    else:
        z = np.column_stack((xNonLin, xNonLin ** 2 - 1, xNonLin ** 3 - 3 * xNonLin))

    return share, xLin, xNonLin, z, Di, mkID, np.repeat(ns, M).reshape((-1, 1)), theta, sigma, pi, R
