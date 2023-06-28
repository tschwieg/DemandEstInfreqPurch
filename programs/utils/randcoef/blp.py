#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function
from numpy.random import normal as randn
from scipy.optimize import fmin
from time import time
import numpy as np
# import timeit

# from utils.randcoef.common import meanval, mdot
from .common import meanval, mdot


def estimateBLP(share,
                xLin,
                xNonLin,
                z,
                mkID,
                ns,
                R,
                Di = None,
                feIndex = None,
                feTransform = None,
                contractionMethod = "squarem",
                contractionIteration = 3,
                contractionUpdate = False,
                boundsAlgorithm = None,
                debug = (),
                debugArray = (),
                debugLevel = 0,
                **kwargs):
    """BLP (1995) estimator for the rand coef logit model

    This is a workworse function to be called via rclogit

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
    contractionMethod : str
        - "squarem": Use the SQUAREM to speed up the δ_{jt} contraction mapping
                     (Varadhan and Roland, 2008)
        - "iteration": Use a regular fixed-point iteration to compute the
                       δ_{jt} contraction mapping
    debug : str or iterable
        A str set to or an iterable containing any of the following:
        - "theta": uses the corresponding entry of debugArray as the true
                  (known) vector of standard deviations for the random coefs
                  and demographic transition matrix. Only optimizes over the
                  linear parameters.
        - "beta0": uses the corresponding entry of debugArray as the starting
                   vector for the linear parameters.
    debugArray : numpy.ndarray or iterable
        The values corresponding to the debug option. It debug is a string
        this must be a numpy array; if it is a list then this must be an
        iterable of numpy arrays.
    debugLevel : int
        Print debug info

    Returns
    -------

    This function returns estimates for β_0, diag(Σ_0) in the model

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

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.
    """

    global betaBLP

    if contractionMethod not in ["squarem", "iteration"]:
        raise Warning("contractionMethod '{0}' not known.".format(contractionMethod))

    # BLP approach
    # ------------

    n  = len(mkID)
    M  = len(np.unique(mkID))

    # Selected data
    # -------------

    seleInd     = (share.flatten() > 0)
    xLinSele    = xLin[seleInd]
    xNonLinSele = xNonLin[seleInd]
    shareSele   = share[seleInd]
    mkIDSele    = mkID[seleInd]
    zSele       = z[seleInd]

    # v_i: Simulate Random coefficients
    # ---------------------------------

    n, k1 = xLinSele.shape
    n, k2 = xNonLinSele.shape
    vi    = randn(0, 1, (k2, R))

    # IV helper
    # ---------

    invZZ  = np.linalg.pinv(zSele.T.dot(zSele))
    xz     = xLinSele.T.dot(zSele)
    invXZZ = np.linalg.pinv(mdot(xz, invZZ, xz.T))
    invXZ  = mdot(invXZZ, xz.dot(invZZ))

    # Starting value of β_0
    # ---------------------

    share0_gen = (shareSele[mkIDSele == t].sum() for t in range(M))
    share0     = (1 - np.fromiter(share0_gen, float))[mkIDSele]
    ySele      = np.log(shareSele / share0.reshape((-1, 1)))
    if feTransform is not None:
        ySele -= feTransform.dot(ySele)[feIndex]
    beta0      = mdot(invXZ, zSele.T.dot(ySele))

    # Special parsing for debug options
    # ---------------------------------

    elements = 0
    debugOK  = True
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

    deltaSele  = xLinSele.dot(beta0).reshape((-1, 1))
    exp_delta0 = np.exp(deltaSele)

    # Run the optimization
    # --------------------

    argsBLP = (shareSele,
               xLinSele,
               xNonLinSele,
               exp_delta0,
               vi,
               zSele,
               invZZ,
               invXZ,
               mkIDSele,
               M,
               contractionMethod,
               contractionIteration,
               contractionUpdate)

    if Di is not None:
        argsBLP += (Di,)
    else:
        argsBLP += (None,)

    if feTransform is not None:
        argsBLP += (feIndex, feTransform, )
    else:
        argsBLP += (None, None, )

    kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}
    kwargsOptim.update(kwargs)

    global debugIter, debugCoefs
    debugIter  = 0
    debugCoefs = theta
    if debugLevel > 0:
        argsBLP += (debugLevel,)
        print("Random coefficients logit model (BLP estimator; debug level {0})".format(debugLevel))
        if contractionMethod == "iteration":
            print("WARNING: contractionMethod = 'iteration' may slow down the computation")

    if contractionUpdate:
        betaBLP = beta0

    if debugLevel > 0:
        timerBase = time()

    if "theta" in debug:
        objectiveBLP(theta, *argsBLP)
        thetaBLP = theta
    else:
        thetaBLP = fmin(func = objectiveBLP,
                        args = argsBLP,
                        x0   = theta,
                        **kwargsOptim)

    if debugLevel > 0:
        timerDelta = (time() - timerBase) / 60
        timerMsg   = "BLP estimator ran in {0:,.2f} minutes"
        print(timerMsg.format(timerDelta))

    if pi is None:
        return thetaBLP.flatten(), betaBLP.flatten()
    else:
        piBLP    = thetaBLP[k2:].reshape((k2, kd))
        sigmaBLP = thetaBLP[:k2]
        return sigmaBLP.flatten(), piBLP, betaBLP.flatten()


def objectiveBLP(theta,
                 share,
                 xLin,
                 xNonLin,
                 exp_delta0,
                 vi,
                 z,
                 invZZ,
                 invXZ,
                 mkID,
                 M,
                 contractionMethod = "squarem",
                 contractionIteration = 3,
                 contractionUpdate = False,
                 Di = None,
                 feIndex = None,
                 feTransform = None,
                 debugLevel = 0):
    """BLP objective J(θ) = ω(θ)' Z (Z' Z)^-1 Z' ω(θ)

    Args
    ----

    theta : numpy.ndarray
        diagnonal elements of Σ, the standard deviation of rand coefs
    share : numpy.ndarray
        Empirical market shares s_{jt}
    xLin : numpy.ndarray
        x_{jt} observables that enter the objective linearly
    xNonLin : numpy.ndarray
        x_{jt} observables that enter the objective non-linearly
    exp_delta0 : numpy.ndarray
        exp(δ_{jt})
    v : numpy.ndarray
        v_i, random coefficients
    z : numpy.ndarray
        Set of instruments Z
    invZZ : numpy.ndarray
        auxiliary array; (Z' Z)^-1 weight matrix
    invXZ : numpy.ndarray
        auxiliary array; [X' Z (Z' Z)^-1 Z' X]^-1 X' Z (Z' Z)^-1
    mkID : numpy.ndarray
        market ID
    M : int
        number of markets
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
    global betaBLP, debugIter, debugCoefs

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
        exp_delta0 = np.exp(xLin.dot(betaBLP).reshape((-1, 1)))

    # Estimate δ_{jt} using iterative prodedure, similar to BLP.
    #     - "squarem" uses the procedure proposed by Varadhan and Roland (2008)
    #       to speed up fixed-point iterations.
    #     - "iteration" uses the traditional fixed-point tieration
    #       that is typically found in BLP.
    if debugLevel > 3:
        delta = meanval(exp_delta0, share, exp_x, mkID, M,
                        contractionMethod, contractionIteration,
                        disp = True, disp_every = 50)
    else:
        delta = meanval(exp_delta0, share, exp_x, mkID, M,
                        contractionMethod, contractionIteration)

    if debugLevel > 2:
        timerDelta1 = time() - timerBase
        timerBase   = time()

    if feTransform is not None:
        delta -= feTransform.dot(delta)[feIndex]

    # Obtain GLS estimate of θhat
    deltaz  = z.T.dot(delta)
    betaBLP = mdot(invXZ, deltaz)

    # Compute ξ_{jt} as the residuals from the model
    xi  = delta - xLin.dot(betaBLP)
    xiz = xi.T.dot(z)

    # Return the objective J(θ)
    J = float(mdot(xiz, invZZ, xiz.T))

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
            debugMsg = r"\t{0}: J = {1:,.6g}, ||Δθ|| = {2:.6g}. Timer ({3}): δ {4:,.1f}, J {5:,.1f}"
            print(debugMsg.format(debugIter, J, diff, timerUnits, timerDelta1, timerDelta2))
        else:
            debugMsg = r"\t{0}: J = {1:,.6g}, ||Δθ|| = {2:.6g}"
            print(debugMsg.format(debugIter, J, diff))

    return J
