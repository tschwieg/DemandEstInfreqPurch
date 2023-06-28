#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from scipy.optimize import fmin, least_squares, basinhopping
from numpy.random import normal as randn
from scipy.stats import norm as normal
from time import time
import numpy as np
# import timeit

# from utils.randcoef.common import meanval, mdot
from .common import meanval, mdot

def estimateBounds(share,
                   xLin,
                   xNonLin,
                   z,
                   mkID,
                   ns,
                   R,
                   Di = None,
                   feIndex = None,
                   feTransform = None,
                   rT = 50,
                   iota = 1e-10,
                   startMethod = "Q",
                   contractionMethod = "squarem",
                   contractionIteration = 3,
                   contractionUpdate = False,
                   boundsKnitro = False,
                   boundsAlgorithm = "least_squares",
                   boundsAlgorithmReps = 5,
                   boundsAlgorithmIter = 100,
                   debug = (),
                   debugArray = (),
                   debugLevel = 0,
                   **kwargs):
    """Gandhi et al. (2017) bounds estimator for the rand coef model

    This is a workworse function that expects to be called from rclogit with
    method = 'bounds', which takes at input a pandas data frame and column
    names. The data passed to this function MUST be sorted. Due to the highly
    non-linear nature of the system, fixed effects are discouraged (rclogit
    will compute the within estimator when fixed effects are requested).

    Args
    ----

    share : numpy.ndarray
        s_{jt}, product market shares
    xLin : numpy.ndarray
        x^1_{jt}, product chars that enter utility linearly
    xNonLin : numpy.ndarray
        x^2_{jt}, product chars that enter utility non-linearly
    z : numpy.ndarray
        z_{jt}, instruments for xLin
    mkID : numpy.ndarray
        Market id
    ns : numpy.ndarray
        Number of goods per market
    R : int
        Number of samples for the random coefficients for xNonLin

    Kwargs
    ------

    Di : numpy.ndarray
        D_i, demographics if applicable
    rT : int
        Control parameter for constructing the set of instrument variable functions
    iota : float
        Parameter that controls the tightness of the bounds; must be in (0, 1)
    startMethod : str
        (Note: This option is overriden by debug)
        - "Q": Starting linear coefs are obtained by setting δ_{jt} = 0 and optimizing
               Q with respect to the linear parameters, β_0
        - "IV": Starting linear coefs are obtained by solving y_{jt} = x_{jt} β + ξ_{jt}
                instrumented with z_{jt}, y_{jt} = log(s_{jt} / s_{0t}).
        - "flat": Starting coefs are set to 0.
    contractionMethod : str
        - "squarem": Use the SQUAREM to speed up the δ_{jt} contraction mapping
                     (Varadhan and Roland, 2008)
        - "iteration": Use a regular fixed-point iteration to compute the
                       δ_{jt} contraction mapping
    boundsAlgorithm : str
        - "least_squares": Optimize inner loop using scipy's least_squares
          (fast; works well with a moderate number of 0s but poorly as the
          pproportion of 0s grows large)
        - "basinhopping": Optimize inner loop using scipy's basinhopping
          (very slow; works well with a large number of 0s)
    debug : str or iterable
        A str set to or an iterable containing any of the following:
        - "theta": uses the corresponding entry of debugArray as the true
                  (known) vector of standard deviations for the random coefs
                  and demographic transition matrix. Only optimizes over the
                  linear parameters.
        - "beta0": uses the corresponding entry of debugArray as the starting
                   vector for the linear parameters.
    debugArray: numpy.ndarray or iterable
        The values corresponding to the debug option. It debug is a string
        this must be a numpy array; if it is a list then this must be an
        iterable of numpy arrays.
    debugLevel : int
        Print debug info

    Returns
    -------

    This function returns estimates for β_0, diag(Σ_0) in the model

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = x_jt β_0 + x_{jt} Σ_0 v_i + x_{jt} Π_0 D_i + ξ_{jt} + ϵ_{ijt}

    using the methods proposed by Gandhi, Lu, and Shi (2017). They note a
    potentially major problem with the popular model proposed in Berry,
    Levinsohn, and Pakes (1995) when there are a non-trivial number of zeros
    in the data; that is, when a not insignificant number of product-market
    shares are 0.

    Traditional approaches involve replacing the market shares with an
    adjusted share so that 0s are replaced with small positive number, or even
    outright dropping the 0s. Both approaches are problematic and result in
    a biased estimator even for a modest proportion of product-market shares
    with 0s in the data (e.g. 10%). As the number of product-market shares
    with 0s grows, the estimator can be highly biased and even change signs.

    Gandhi et al. deal with this problem by constructing bounds around δ_{ij}
    while making use of the full data. Their estimator appears to perform well
    even in the presence of an extreme number of 0s (> 90%).

    References
    ----------

    - Nevo, A. (2000). A Practitioner’s Guide to Estimation of
      Random-Coefficients Logit Models of Demand. Journal of Economics &
      Management Strategy, 9(4), 513–548.

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.

    - Gandhi, Amit, Zhentong Lu, and Xiaoxia Shi (2017). Estimating Demand for
      Differentiated Products with Zeroes in Market Share Data. Working Paper.
    """

    if contractionMethod not in ["squarem", "iteration"]:
        raise Warning("contractionMethod '{0}' not known.".format(contractionMethod))

    sIV   = startMethod == "IV"
    sQ    = startMethod == "Q"
    sFlat = startMethod == "flat"
    sAny  = sIV or sQ or sFlat
    if not sAny:
        for method in startMethod:
            if method not in ["IV", "Q", "flat"]:
                raise Warning("startMethod '{0}' not known.".format(method))

    global betaBounds

    # Bounds approach
    # ---------------

    n   = len(mkID)
    M   = len(np.unique(mkID))
    Jt  = np.diff(np.concatenate(([0], np.where(np.diff(mkID))[0] + 1, [n]))).reshape((-1, 1))
    J   = Jt.mean()
    eta = ((1 - iota) / (ns + Jt + 1))[mkID]

    # Compute µ(g), the weights for g(), and g(), the dummies 1(z_{jt} in
    # B_g); see gCubes for more.
    mu_g, g = gCubes(z, rT)
    mu_w    = np.row_stack((mu_g, mu_g)).flatten() ** 0.5
    if feTransform is not None:
        g  -= feTransform.dot(g)[feIndex]

    # v_i: Simulate Random coefficients
    # ---------------------------------

    n, k1 = xLin.shape
    n, k2 = xNonLin.shape
    vi    = randn(0, 1, (k2, R))

    # Laplace shares
    # --------------

    shareLap   = (share * ns[mkID] + 1) / (ns + Jt + 1)[mkID]
    share0_gen = (share[mkID == t].sum() for t in range(M))
    share0_tot = ns * (1 - np.fromiter(share0_gen, float).reshape((-1, 1)))
    shareLap0  = ((share0_tot + 1) / (ns + Jt + 1))[mkID]

    # Starting value of β_0
    # ---------------------

    # Not really sure what a good way of setting the starting value for
    # δ_{jt} is... In the BLP case, we simply solve the GLS problem with
    # y = log(s_{jt} / s_{0t}), but here it is not clear.

    boundsAlgorithmGlobal = (boundsAlgorithm == "basinhopping")
    if "IV" in startMethod:

        y = np.log(shareLap / shareLap0)
        if feTransform is not None:
            y -= feTransform.dot(y)[feIndex]

        invZZ  = np.linalg.pinv(z.T.dot(z))
        xz     = xLin.T.dot(z)
        invXZZ = np.linalg.pinv(mdot(xz, invZZ, xz.T))
        invXZ  = mdot(invXZZ, xz.dot(invZZ))
        beta0  = mdot(invXZ, z.T.dot(y)).flatten()

    if "Q" in startMethod:

        shareRatio = shareLap0 / shareLap
        delta_u = np.log(shareRatio * ((shareLap + eta) / (shareLap0 - eta)))
        delta_l = np.log(shareRatio * ((shareLap - eta) / (shareLap0 + eta)))
        if feTransform is not None:
            delta_u -= feTransform.dot(delta_u)[feIndex]
            delta_l -= feTransform.dot(delta_l)[feIndex]

        # Maux    = 1 / M ** 0.5
        Maux    = 1 / (M * J)
        Xaux    = g.T.dot(xLin)
        Y       = Maux * np.row_stack((g.T.dot(-delta_l), g.T.dot(delta_u))).flatten()
        X       = Maux * np.row_stack((Xaux, -Xaux))

        if "IV" not in startMethod:
            # beta0 = np.array([0 for i in range(k1)])
            beta0 = np.array([1 for i in range(k1)])

        if False:
            beta0 = np.array([1 for i in range(k1)])
        elif boundsAlgorithmGlobal:
            # if boundsAlgorithmGlobal:
            # NOTE(mauricio): The loop should be superfluous if the basinhopping
            # call is specified correctly; read up on this. // 2017-08-28 09:02 EDT
            i   = 0
            new = True
            optimQ = basinhopping(func = basinhoppingQ, x0 = beta0,
                                  minimizer_kwargs = {"args": (Y, X, mu_w)},
                                  niter = boundsAlgorithmIter)
            while (i < boundsAlgorithmReps) & (new):
                Qbounds    = optimQ.fun
                betaBounds = optimQ.x
                optimQ     = basinhopping(func = basinhoppingQ, x0 = betaBounds,
                                          minimizer_kwargs = {"args": (Y, X, mu_w)},
                                          niter = boundsAlgorithmIter)

                new = optimQ.fun < Qbounds
                i  += 1
        else:
            optimQ = least_squares(objectiveQ, x0 = beta0, args = (Y, X, mu_w))

        beta0  = optimQ.x

    if "flat" in startMethod:
        beta0 = np.array([0 for i in range(k1)])

    # Special parsing for debug options
    # ---------------------------------

    elements = 0
    debugOK  = True
    # sigma    = np.array([0 for i in range(k2)])
    sigma    = np.array([1 for i in range(k2)])

    if Di is not None:
        kd, nd = Di.shape
        pi     = np.ones((k2, kd))
        theta  = np.concatenate((sigma, pi.flatten()))
    else:
        kd    = None
        pi    = None
        theta = sigma

    if "theta" in debug:
        if "theta" == debug:
            theta   = debugArray
            debugOK = True
        else:
            elements += 1
            debugOK   = False

    if "beta0" in debug:
        if "beta0" == debug:
            beta0   = debugArray
            debugOK = True
        else:
            elements += 1
            debugOK   = False

    if elements > 0:
        for dVal, dArr in zip(debug, debugArray):
            if dVal == "theta":
                theta   = dArr
                debugOK = True
            elif dVal == "beta0":
                beta0   = dArr
                debugOK = True

    if not debugOK:
        raise Warning("debug or debugArray was not correctly specified.")

    # Starting value of δ_{jt}
    # ------------------------

    delta      = xLin.dot(beta0).reshape((-1, 1))
    exp_delta0 = np.exp(delta)

    # Run the optimization
    # --------------------

    argsBounds = (shareLap,
                  shareLap0,
                  xLin,
                  xNonLin,
                  exp_delta0,
                  vi,
                  g,
                  mu_w,
                  mkID,
                  M,
                  J,
                  eta,
                  beta0,
                  boundsKnitro,
                  boundsAlgorithmGlobal,
                  boundsAlgorithmReps,
                  boundsAlgorithmIter,
                  contractionMethod,
                  contractionIteration,
                  contractionUpdate)

    if Di is not None:
        argsBounds += (Di,)
    else:
        argsBounds += (None,)

    if feTransform is not None:
        argsBounds += (feIndex, feTransform, )
    else:
        argsBounds += (None, None, )

    kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}
    kwargsOptim.update(kwargs)

    global debugIter, debugCoefs
    debugIter  = 0
    debugCoefs = theta
    if debugLevel > 0:
        argsBounds += (debugLevel,)
        print("Random coefficients logit model (bounds estimator; debug level {0})".format(debugLevel))
        if contractionMethod == "iteration":
            print("WARNING: contractionMethod = 'iteration' may slow down the computation")

    if contractionUpdate:
        betaBounds = beta0

    if debugLevel > 0:
        timerBase = time()

    if "theta" in debug:
        objectiveBounds(theta, *argsBounds)
        thetaBounds = theta
    else:
        if False:
            beta0 = np.array([1 for i in range(k1)])
        elif boundsAlgorithmGlobal:
            thetaBounds = basinhopping(func = objectiveBounds,
                                       minimizer_kwargs = {"args": argsBounds},
                                       x0   = theta,
                                       **kwargsOptim)
        else:
            thetaBounds = fmin(func = objectiveBounds,
                               args = argsBounds,
                               x0   = theta,
                               **kwargsOptim)

    if debugLevel > 0:
        timerDelta = (time() - timerBase) / 60
        timerMsg   = "Bounds estimator ran in {0:,.2f} minutes"
        print(timerMsg.format(timerDelta))

    if pi is None:
        return thetaBounds.flatten(), betaBounds.flatten()
    else:
        piBounds    = thetaBounds[k2:].reshape((k2, kd))
        sigmaBounds = thetaBounds[:k2]
        return sigmaBounds.flatten(), piBounds, betaBounds.flatten()


def gCubes(zContinuous, rT):
    """g() instrument function

    This function gives the instruments g(z_{jt}) in G; we take the set of
    dummies 1(z_{jt} in B_g) for B_g a hybercube in the support of z. It also
    gives the corresponding weights, µ(g).

    Per Gandhi et al. (2017), we normalize the set of instruments to

        zbar(z) = F_{N(0, 1)} ( Σ^{-0.5}_z (z - μ_z) )

    whre F_{N(0, 1)} is the CDF of a standard normal. The set G is then
    defined as

        G = {g(z_c, z_d) = 1(zbar(z_c), z_d) in C}
        C in {cross 1 to K_c ((a - 1) / 2r, a / 2r)} cross {ς}
        K_c the number of continuous instruments
        Z_d the set of values discrete instruments can take
        z_c continuous instruments
        z_d discrete instruments in Z_d
        ς in Z_d
        a in {1, 2, ..., 2r}
        r in {r0, r0 + 1, ...}

    In practice, r in {r0, r0 + 1, ..., r0 + rT}. We also compute μ(g) the
    weights for the set of IV functions:

        μ(g) ∝ (100 + r)^{-2} (2r)^{-d} k_d^{-1}

    for g in G, where d is the number of instruments and K_d is the number of
    elements in Z_d.
    """

    # here z is just one-dimensional, so k=1
    n, kc = zContinuous.shape

    # kd should be set of possible values, not the number of variabes
    # kd
    # zDiscrete
    kd = 1

    # transform zContinuous into unit interval
    muZ    = zContinuous.mean(0)
    sigmaZ = zContinuous.std(0)
    zBar   = normal.cdf((zContinuous - muZ) / sigmaZ)

    # number of weight functions
    r1  = np.arange(rT) + 1
    n_g = rT * (rT + 1) * kc

    # weights for g()
    # wr = 1 / rT
    wr = 1 / (100 + r1 ** 2)
    wa = 1 / (kd * kc * 2 * r1)
    mu = wr * wa

    # construct g functions: indicators
    r0 = 1
    i  = 0
    k  = kc
    g  = np.zeros((n, n_g))
    mu_g  = np.zeros((n_g + 1, 1))
    mu_to = 0

    # weights on the constant is equal to some constant bigger than the weight
    # of the coarsest hypercube
    mu_g[-1] = 1 / (kd * kc * 2 * 101)

    # Transform the instruments zContinuous into g(z), an indicator of whether
    # z is in hypercube in kc dimensions whose vertices are ((a - 1) / 2r, a / 2r)
    # for a = 1, ..., 2r and r = 1, ..., rT

    for r in range(r0, rT + 1):
        mu_from  = mu_to
        mu_to   += kc * 2 * r
        mu_g[mu_from:mu_to] = mu[r - 1]
        for a in range(1, 2 * r + 1):
            indFunc   = (zBar > (a - 1) / (2 * r)) & (zBar <= a / (2 * r))
            g[:, i:k] = indFunc.astype(int)
            i += kc
            k += kc

    return mu_g, np.column_stack((g, np.ones((n, 1))))


# def gBoxes(zContinuous, rT):
#     n, kc  = zContinuous.shape
#     kd     = 1
#     muZ    = zContinuous.mean(0)
#     sigmaZ = zContinuous.std(0)
#     zBar   = normal.cdf((zContinuous - muZ) / sigmaZ)


def objectiveBounds(theta,
                    shareLap,
                    shareLap0,
                    xLin,
                    xNonLin,
                    exp_delta0,
                    vi,
                    g,
                    mu_w,
                    mkID,
                    M,
                    J,
                    eta,
                    beta0,
                    boundsKnitro = False,
                    boundsAlgorithmGlobal = False,
                    boundsAlgorithmReps = 5,
                    boundsAlgorithmIter = 100,
                    contractionMethod = "squarem",
                    contractionIteration = 3,
                    contractionUpdate = False,
                    Di = None,
                    feIndex = None,
                    feTransform = None,
                    debugLevel = 0):
    """Bounds objective, min_β Q_T(β; Σ)

    Args
    ----

    theta : numpy.ndarray
        diagnonal elements of Σ, the standard deviation of rand coefs, and vec(Π), if provided.
    shareLap : numpy.ndarray
        Laplace shares (s_{jt} * n_t + 1) / (n_t + J_t + 1)
    shareLap0 : numpy.ndarray
        Laplace share of outside good, s_{0t}
    xLin : numpy.ndarray
        x_{jt} observables that enter the objective linearly
    xNonLin : numpy.ndarray
        x_{jt} observables that enter the objective non-linearly
    exp_delta0 : numpy.ndarray
        exp(δ_{jt})
    vi : numpy.ndarray
        v_i, random coefficients
    g : numpy.ndarray
        Set of dummies 1(z_{jt} in B_g) in G, for B_g a hybercube in the support of z
    mu_w : numpy.ndarray
        µ(g)^0.5, square root of the corresponding weights
    mk_id : numpy.ndarray
        market ID
    M : int
        number of markets
    eta : float
        η_t, parameter to control the width of the bounds
    beta0 : numpy.ndarray
        Initial guess for linear parameters
    contractionMethod : str
        - "squarem": Use the SQUAREM to speed up the δ_{jt} contraction mapping
                     (Varadhan and Roland, 2008)
        - "iteration": Use a regular fixed-point iteration to compute the
                       δ_{jt} contraction mapping

    Kwargs
    ------

    Di : numpy.ndarray
        D_i, demographics if applicable
    debugLevel : int
        Print debug info

    Returns
    -------

    The bounds objective, namely

        min_β Q_T(β; Σ)

    where

        Q_T(θ) = ∑_{g in G} µ(g) ([ρ^u_T(θ, g)]_^2 + [ρ^l_T(θ, g)]_^2),

    θ = (β; Σ), []_ is a funciton s.t. [a]_ = min(0, a), and

        ρ^u_T(θ, g) = ∑_{j, t} [δ^u_{jt}(Σ) - x_{jt} β] * g(z_jt)
        ρ^l_T(θ, g) = ∑_{j, t} [x_{jt} β - δ^l_{jt}(Σ)] * g(z_jt)

    Gandhi et a. propose a modification to the standard random-coefficients
    approach. Typically we have

        θ = argmin J(θ)
        J(θ) = ω(θ)' Z (Z' Z)^-1 Z' ω(θ)

    with ω(θ) = (δ_{jt}(Σ) - x_{jt} β). Here we model

        δ_{jt} = x_{jt} β + ξ_{jt}

    and identification comes from an IV approach, that is, for some set of
    instruments z_{jt} we have

        E[ξ_{jt} | z_{jt}] = 0

    However, in the presence of 0s for market shares s_{jt} we would need

        E[ξ_{jt} | z_{jt}, s_{jt} > 0] = 0

    but this is generally not the case. If we keep the 0s, we know that

        E[δ_{jt} - x_{jt} β | z_{jt}] = 0

    Hence Gandhi et al. propose constructing bounds on this expectation,
    that is

        E[δ^u_{jt} - x_{jt} β | z_{jt}] >= 0 >= E[δ^l_{jt} - x_{jt} β | z_{jt}]

    They then construct moment conditions by using a space of instrumental
    variable functions G so that

        E[(δ^u_{jt} - x_{jt} β) g(z_{jt})] >= 0 >= E[(δ^l_{jt} - x_{jt} β) g(z_{jt})]

    (see the paper for details on G). The bounds are constructed as

        δ^u_{jt} = Δ_{jt} + log (s'_{jt} + η_t) - log (s'_{0t} - η_t)
        δ^l_{jt} = Δ_{jt} + log (s'_{jt} - η_t) - log (s'_{0t} + η_t)
        Δ_{jt}   = δ_{jt} - log (s'_{jt}) - log (s'_{0t})
        η_t      = (1 - ι) / (n_t + J_t + 1)

    where s'_{jt} are the laplace shares, (n s_{jt} + 1) / (n + J + 1), and
    ι is chosen to be reasonably small (ι must be in the unit interval, so
    Gandhi et al. suggest starting with 10e-3 and dividing by 10 until the
    estiamtes stabilize). For intuition, consider the case when Δ_{jt} = 0.
    The bounds above simply perturb the ratio of the Laplace shares away from
    0 in either direction.

    Once the bounds have been constructed, we can proceed normally by
    optimizing Q_T(β; Σ) (which can be done in two steps; much like the
    standard BLP estimator, we can express β ≡ β(Σ)). For detais on the
    consistency of this estimator, see Gandhi et al.

    References
    ----------

    - Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices In
      Market Equilibrium. Econometrica, 63(4), 841.

    - Nevo, A. (2000). A Practitioner’s Guide to Estimation of
      Random-Coefficients Logit Models of Demand. Journal of Economics &
      Management Strategy, 9(4), 513–548.

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.
    """
    global betaBounds, debugIter, debugCoefs

    if debugLevel > 2:
        timerBase = time()

    # exp(x_{jt} Σ_0 v_i)
    if Di is not None:
        kd, nd = Di.shape
        kv, nv = vi.shape
        pi     = theta[kv:].reshape((kv, kd))
        sigma  = theta[:kv]
        exp_x  = np.exp(xNonLin.dot(sigma[:, None] * vi + pi.dot(Di)))
    else:
        exp_x  = np.exp(xNonLin.dot(theta[:, None] * vi))

    if contractionUpdate:
        exp_delta0 = np.exp(xLin.dot(betaBounds).reshape((-1, 1)))

    # Estimate δ_{jt} using iterative prodedure, similar to BLP.
    #     - "squarem" uses the procedure proposed by Varadhan and Roland (2008)
    #       to speed up fixed-point iterations.
    #     - "iteration" uses the traditional fixed-point tieration
    #       that is typically found in BLP.
    if debugLevel > 3:
        delta = meanval(exp_delta0, shareLap, exp_x, mkID, M,
                        contractionMethod, contractionIteration,
                        disp = True, disp_every = 50)
    else:
        delta = meanval(exp_delta0, shareLap, exp_x, mkID, M,
                        contractionMethod, contractionIteration)

    if debugLevel > 2:
        timerDelta1 = time() - timerBase
        timerBase   = time()

    # if feTransform is not None:
    #     delta -= feTransform.dot(delta)[feIndex]

    # Compute bounds for δ_{jt} following Gandhi et al. (2017)
    shareRatio = shareLap0 / shareLap
    delta_u    = delta + np.log(shareRatio * ((shareLap + eta) / (shareLap0 - eta)))
    delta_l    = delta + np.log(shareRatio * ((shareLap - eta) / (shareLap0 + eta)))

    if feTransform is not None:
        delta_u -= feTransform.dot(delta_u)[feIndex]
        delta_l -= feTransform.dot(delta_l)[feIndex]

    # Compute the elements of ρ^l_T(θ, g) and ρ^u_T(θ, g); in this case
    # both are sums over j, t of (δ_{jt} - x_{jt} β) g(z_{jt}). and we
    # compute
    #
    #     a) ∑_{j, t} -δ^l_{jt} * g(z_jt)
    #     b) ∑_{j, t}  δ^u_{jt} * g(z_jt)
    #     c) ∑_{j, t}   -x_{jt} * g(z_jt)
    #     d) ∑_{j, t}    x_{jt} * g(z_jt)
    #
    # We will later compute min(a - c β, 0) and min(b - d β, 0)

    # Maux  = 1 / M ** 0.5
    # Y_sum = g.T.dot(np.column_stack((-delta_l, delta_u)))  # a, b
    # X_sum = g.T.dot(np.column_stack((-x_lin, x_lin)))      # c, d
    Maux  = 1 / (M * J)
    Xaux  = g.T.dot(xLin)
    Y     = Maux * np.row_stack((g.T.dot(-delta_l), g.T.dot(delta_u))).flatten()
    X     = Maux * np.row_stack((Xaux, -Xaux))

    # Return the objective (minimize Q wrt β; see estimateQ for more)
    if False:
        Qbounds    = basinhoppingQ(betaBounds, Y, X, mu_w)
    elif boundsAlgorithmGlobal:
        # if boundsAlgorithmGlobal:
        # NOTE(mauricio): The loop should be superfluous if the basinhopping
        # call is specified correctly; read up on this. // 2017-08-28 09:02 EDT
        i   = 0
        new = True
        optimQ = basinhopping(func = basinhoppingQ, x0 = beta0,
                              minimizer_kwargs = {"args": (Y, X, mu_w)},
                              niter = boundsAlgorithmIter)
        while (i < boundsAlgorithmReps) & (new):
            Qbounds    = optimQ.fun
            betaBounds = optimQ.x
            optimQ     = basinhopping(func = basinhoppingQ, x0 = betaBounds,
                                      minimizer_kwargs = {"args": (Y, X, mu_w)},
                                      niter = boundsAlgorithmIter)

            new = optimQ.fun < Qbounds
            i  += 1

    else:
        optimQ     = least_squares(objectiveQ, x0 = beta0, args = (Y, X, mu_w))
        Qbounds    = optimQ.cost
        betaBounds = optimQ.x

    if debugLevel > 2:
        timerDelta2 = time() - timerBase
        if min(timerDelta1, timerDelta2) > 60:
            timerDelta1 /= 60
            timerDelta2 /= 60
            timerUnits   = "min"
        else:
            timerUnits   = "sec"

    if debugLevel > 1:
        debugIter += 1
        diff       = (((theta - debugCoefs) ** 2).sum()) ** 0.5
        debugCoefs = theta
        if debugLevel > 2:
            debugMsg  = r"\t{0}: min Q = {1:,.6g}, ||Δθ|| = {2:.6g}. Timer ({3}): δ {4:,.1f}, Q {5:,.1f}"
            print(debugMsg.format(debugIter, Qbounds, diff, timerUnits, timerDelta1, timerDelta2))
        else:
            debugMsg  = r"\t{0}: min Q = {1:,.6g}, ||Δθ|| = {2:.6g}"
            print(debugMsg.format(debugIter, Qbounds, diff))

    return Qbounds


def objectiveQ(b, Y, X, mu_w):
    """Compute the elements of Q for min_β Q_T(β; Σ)

    Recall

        Q_T(β; Σ) = ∑_{g in G} µ(g) ([ρ^u_T(θ, g)]_^2 + [ρ^l_T(θ, g)]_^2)

    where []_ is a funciton where [a]_ = min(0, a). Y and X are the components
    of ρ^u_T(θ, g) and ρ^l_T(θ, g); the least_squares routine squares and
    sums the elements of the vector returned by objectiveQ, so we simply need
    to return the individual elements of the functions.

    Y, X that we pass to this function are already summed over j, t, so we
    only need to times X by the current estimate for the linear parameters and
    subtract from the current estimates for -δ^l_{jt} and δ^u_{jt}, which
    are in Y. mu_w contains µ(g)^0.5 so that when squared each element gets
    multiplied by its weight when squared.
    """
    return np.minimum(Y + X.dot(b), np.zeros(Y.shape)) * mu_w


def basinhoppingQ(b, Y, X, mu_w):
    return (objectiveQ(b, Y, X, mu_w) ** 2).sum()
