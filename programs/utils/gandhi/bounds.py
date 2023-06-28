# -*- coding: utf-8 -*-

from __future__ import division, print_function
from scipy.optimize import fmin, least_squares
from numpy.random import normal as randn
from scipy.stats import norm as normal
from os import linesep
import numpy as np

from .common import meanval, mdot

def estimateBounds(share,
                   x_lin,
                   x_nonlin,
                   z,
                   mk_id,
                   ns,
                   R,
                   r_n = 50,
                   iota = 1e-10,
                   alt = False,
                   boostrap = 0,
                   d0 = [1],
                   theta0 = None):
    """Gandhi et al. (2017) bounds estimator for the rand coef model

    Args
    ----

    share (numpy.ndarray): s_{jt}, prod market shares
    x_lin (numpy.ndarray): x^1_{jt}, prod chars that enter utility linearly
    x_nonlin (numpy.ndarray): x^2_{jt}, prod chars that enter utility non-linearly
    z (numpy.ndarray): z_{jt}, instruments for x_lin
    mk_id (numpy.ndarray): Market id
    ns (numpy.ndarray): Number of goods per market
    R (int): Number of samples for the random coefficients for x_nonlin

    Kwargs
    ------

        iota (int): Control parameter for constructing the instruments
        iota (float): Parameter that controls the tightness of the bounds
        alt (bool): whether to pass the true (known) standard deviation matrix
                    for the random coefficients to the objective.
        boostrap (int): Repetitions for boostrap standard errors (0 means no SE)

    Returns
    -------

    This function returns estimates for β_0, Σ_0 in the model

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = x_jt β_0 + x_{jt} Σ_0 v_i + ξ_{jt} + ϵ_{ijt}

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

    - Gandhi, Amit, Zhentong Lu, and Xiaoxia Shi (2017). Estimating Demand for
      Differentiated Products with Zeroes in Market Share Data. Working Paper.
    - Nevo, A. (2000). A Practitioner’s Guide to Estimation of
      Random-Coefficients Logit Models of Demand. Journal of Economics &
      Management Strategy, 9(4), 513–548.
    """
    global beta

    # Bounds approach
    # ---------------

    M   = len(np.unique(mk_id))
    J   = int(np.unique(np.diff(np.where(np.diff(mk_id)))))
    eta = (1 - iota) / (ns + J + 1)

    # Compute µ(g), the weights for g(), and g(), the dummies 1(z_{jt} in
    # B_g); see g_func for more.
    mu_g, g1 = g_func(z, r_n)
    g = np.column_stack((g1, np.ones((g1.shape[0], 1))))

    # weights on the constant is equal to some constant bigger than the weight
    # of the coarsest hypercube of the x
    mu_g[g.shape[1] - 1] = 1 / 202

    # v_i: Simulate Random coefficients
    # ---------------------------------

    n, k1 = x_lin.shape
    n, k2 = x_nonlin.shape
    v     = randn(0, 1, (k2, R))

    # Laplace shares
    # --------------

    dummies    = np.row_stack((mk_id == t for t in range(M))).astype(int)
    share_lap  = (share * ns + 1) / (ns + J + 1)
    out_ch_aux = ns - dummies.dot(share * ns)
    out_ch     = np.array([t for j in range(J) for t in out_ch_aux])
    share_lap0 = (out_ch + 1) / (ns + J + 1)

    # Not really sure what a good way of setting the starting value for
    # δ_{jt} is... In the BLP case, the first (d0 = 1) is used. However,
    # here would it might make more sense to use ths second?

    sigma = np.array([1 for i in range(k2)])
    if theta0 is not None:
        beta0 = theta0

    if 0 in d0:

        # IV helper
        # ---------

        # y      = np.log(share_lap / share_lap0)
        # invZZ  = np.linalg.pinv(g.T.dot(g))
        # xz     = x_lin.T.dot(g)
        # invXZZ = np.linalg.pinv(mdot(xz, invZZ, xz.T))
        # invXZ  = mdot(invXZZ, xz.dot(invZZ))
        # beta0  = mdot(invXZ, g.T.dot(y)).flatten()

        y      = np.log(share_lap / share_lap0)
        invZZ  = np.linalg.pinv(z.T.dot(z))
        xz     = x_lin.T.dot(z)
        invXZZ = np.linalg.pinv(mdot(xz, invZZ, xz.T))
        invXZ  = mdot(invXZZ, xz.dot(invZZ))
        beta0  = mdot(invXZ, z.T.dot(y)).flatten()

    if 1 in d0:

        # Starting value of ξ_{jt}
        # ------------------------

        share_ratio = share_lap0 / share_lap
        delta_u = np.log(share_ratio * ((share_lap + eta) / (share_lap0 - eta)))
        delta_l = np.log(share_ratio * ((share_lap - eta) / (share_lap0 + eta)))
        Y_sum   = g.T.dot(np.column_stack((-delta_l, delta_u))) 
        X_sum   = g.T.dot(np.column_stack((-x_lin, x_lin)))     
        Maux    = 1 / M ** 0.5
        if 0 not in d0:
            beta0 = np.array([0 for i in range(k1)]) if theta0 is None else theta0

        # Y = Maux * Y_sum
        # X = Maux * X_sum
        estimateQ(Maux * Y_sum, Maux * X_sum, beta0, mu_g)
        beta0 = beta

    if 2 in d0:
        beta0 = np.array([0 for i in range(k1)]) if theta0 is None else theta0

    # theta0    = np.array([-13, 1])
    delta     = x_lin.dot(beta0).reshape((-1, 1))
    exp_delta = np.exp(delta)

    # Actual optimization
    # -------------------

    argsBounds = (share_lap,
                  share_lap0,
                  x_lin,
                  x_nonlin,
                  exp_delta,
                  v,
                  g,
                  mu_g,
                  mk_id,
                  M,
                  eta,
                  beta0)

    kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}

    beta = np.zeros(beta0.shape)
    if alt:
        sigma_bounds = np.array([0.5])
        objectiveBounds(sigma_bounds, *argsBounds)
    else:
        sigma_bounds = fmin(func = objectiveBounds,
                            args = argsBounds,
                            x0   = sigma,
                            **kwargsOptim)

    beta_bounds = beta
    if boostrap > 0:
        boot = np.empty((0, len(np.concatenate((sigma_bounds, beta)))))
        for b in range(boostrap):
            try:
                bootsample = np.random.randint(n, size = n)
                mk_id_boot = mk_id[bootsample]
                dummies    = np.row_stack((mk_id_boot == t for t in range(M))).astype(int)
                share_boot = (share[bootsample] * ns + 1) / (ns + J + 1)
                out_ch_aux = ns - dummies.dot(share * ns)
                out_ch     = np.array([t for j in range(J) for t in out_ch_aux])
                share_lap0 = (out_ch + 1) / (ns + J + 1)

                argsBoot   = (share_boot,
                              share_lap0[bootsample],
                              x_lin[bootsample],
                              x_nonlin[bootsample],
                              exp_delta[bootsample],
                              v,
                              g[bootsample],
                              mu_g,
                              mk_id_boot,
                              M,
                              eta,
                              beta0)
                sigma_boot = fmin(func = objectiveBounds,
                                  args = argsBoot,
                                  x0   = sigma,
                                  **kwargsOptim)
                boot = np.row_stack((boot, np.concatenate((sigma_boot, beta))))
                print(".", end = '' if (b + 1) % 50 else linesep)
            except:
                print("x", end = '' if (b + 1) % 50 else linesep)

        ci_l = np.percentile(boot, [2.5], axis = 0)
        ci_u = np.percentile(boot, [97.5], axis = 0)
        se   = boot.std(axis = 0)
        return np.concatenate((sigma_bounds, beta_bounds)), se, ci_l, cu_u
    else:
        return sigma_bounds, beta_bounds


def objectiveBounds(sigma,
                    share_lap,
                    share_lap0,
                    x_lin,
                    x_nonlin,
                    exp_delta,
                    v,
                    g,
                    mu_g,
                    mk_id,
                    M,
                    eta,
                    beta0):
    """Bounds objective (min_Σ) min_β Q_T(β; Σ) 

    Args
    ----

    sigma (numpy.ndarray): diagnonal elements of Σ, the standard deviation of rand coefs
    share_lap (numpy.ndarray): Laplace shares (s_{jt} * n_t + 1) / (n_t + J_t + 1)
    share_lap0 (numpy.ndarray): Laplace share of outside good, s_{0t}
    x_lin (numpy.ndarray): x_{jt} observables that enter the objective linearly
    x_nonlin (numpy.ndarray): x_{jt} observables that enter the objective non-linearly
    exp_delta (numpy.ndarray): exp(δ_{jt})
    v (numpy.ndarray): v_i, random coefficients
    g (numpy.ndarray): Set of dummies 1(z_{jt} in B_g) in G, for B_g a hybercube in the support of z
    mu_g (numpy.ndarray): µ(g), corresponding weights
    mk_id (numpy.ndarray): market ID
    M (int): number of markets
    eta (float): η_t, parameter to control the width of the bounds
    beta0 (numpy.ndarray): Initial guess for linear parameters

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
    """
    global beta

    # exp(x_{jt} Σ_0 v_i)
    x_ind_exp = np.exp((sigma[:, None] * v) * x_nonlin)
    # exp_delta = np.exp(x_lin.dot(beta).reshape((-1, 1)))

    # Estimate δ_{jt} using iterative prodedure, similar to BLP
    # timerBase = time.time()
    # delta_lap  = meanval(exp_delta, share_lap, x_ind_exp, mk_id, M, tol = 1e-14)
    # timerDelta  = (time.time() - timerBase)
    # print("{0:.2f} seconds".format(timerDelta))
    # np.abs(delta_lap - xx.x0).max()
    delta_lap = meanval(exp_delta, share_lap, x_ind_exp, mk_id, M, debug = True, tol = 1e-14)
    delta_lap = meanval(np.exp(delta_lap), share_lap, x_ind_exp, mk_id, M, debug = True, tol = 1e-14)
    delta_lap = meanval(np.exp(xx.x0), share_lap, x_ind_exp, mk_id, M, debug = True)

    # Compute bounds for δ_{jt} following Gandhi et al. (2017)
    share_ratio = share_lap0 / share_lap
    delta_u     = delta_lap + np.log(share_ratio * ((share_lap + eta) / (share_lap0 - eta)))
    delta_l     = delta_lap + np.log(share_ratio * ((share_lap - eta) / (share_lap0 + eta)))

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

    Y_sum = g.T.dot(np.column_stack((-delta_l, delta_u)))  # a, b
    X_sum = g.T.dot(np.column_stack((-x_lin, x_lin)))      # c, d
    Maux  = 1 / M ** 0.5

    # Return the objective (minimize Q wrt β; see estimateQ for more)
    # Y = Maux * Y_sum
    # X = Maux * X_sum
    return estimateQ(Maux * Y_sum, Maux * X_sum, beta0, mu_g)


def g_func(z, r):
    """g() instrument function

    This function gives the instruments g(z_{jt}) in G; we take the set of
    dummies 1(z_{jt} in B_g) for B_g a hybercube in the support of z. It also
    gives the corresponding weights, µ(g).
    """

    # here z is just one-dimensional, so k=1
    n, k = z.shape

    # transform z into unit interval
    z_mu  = np.kron(np.ones((n, 1)), z.mean(0))
    z_std = np.kron(np.ones((n, 1)), z.std(0))
    z     = normal.cdf((z - z_mu) / z_std)

    # number of weight functions
    r2  = np.arange(2, (2 * r + 2), 2).reshape(-1, 1)
    n_g = r2.sum() * k

    # weight function
    mu = 1 / (((r2 / 2 + 100) ** 2) * (k * r2))

    # construct g functions: indicators
    i = 0
    f = np.zeros((n, n_g))
    mu_g = np.zeros((n_g + 1, 1))
    for i1 in range(k):
        for r1 in range(1, r + 1):
            for a1 in range(1, 2 * r1 + 1):
                ind_temp = (z[:, i1] >= (a1 - 1) / (2 * r1)) & (z[:, i1] <= a1 / (2 * r1))
                f[:, i] = ind_temp
                mu_g[i, 0] = mu[r1 - 1]
                i += 1

    return np.array(mu_g), f


def estimateQ(Y, X, beta0, mu_g):
    """Estimate min_β Q_T(β; Σ)

    Recall

        Q_T(β; Σ) = ∑_{g in G} µ(g) ([ρ^u_T(θ, g)]_^2 + [ρ^l_T(θ, g)]_^2)

    where []_ is a funciton where [a]_ = min(0, a). Y and X are the components
    of ρ^u_T(θ, g) and ρ^l_T(θ, g); the least_squares routine squares and
    sums the elements of the vector returned by objectiveQ, so we simply need
    to return the individual elements of the functions.
    """
    # , "max_nfev": 10000
    global beta
    kwargsOptimLsq = {"xtol": 1e-10, "ftol": 1e-10}
    resOptim = least_squares(objectiveQ, x0 = beta0, args = (Y, X, mu_g), **kwargsOptimLsq)
    beta = resOptim.x
    return resOptim.cost

# least_squares(objectiveQ, x0 = beta0,  args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = theta0, args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-13, 0.6]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-13, 0.7]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-13, 0.8]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-13, 0.9]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-13, 1.0]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-12, 1.0]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-11, 1.0]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-10, 1.0]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# least_squares(objectiveQ, x0 = np.array([-9,  1.0]), args = (Y, X, mu_g), **kwargsOptimLsq).x
# fmin(func = test, x0 = beta0,  args = (Y, X, mu_g), **kwargsOptim)
# fmin(func = test, x0 = theta0, args = (Y, X, mu_g), **kwargsOptim)

def test(b, Y, X, mu_g):
    return (objectiveQ(b, Y, X, mu_g) ** 2).sum()


def objectiveQ(b, Y, X, mu_g):
    """Compute the elements of ρ^u_T(θ, g) and ρ^l_T(θ, g)

    The least_squares function automagically squares and sums them. Recall
    that Y, X that we pass to this function are already summed over j, t, so
    we only need to times X by the current estimate for the linear parameters
    and subtract from the current estimates for -δ^l_{jt} and δ^u_{jt}. Note
    we apply the []_ function and take the minimum between each element and
    0, and we times by µ(g)^0.5 so that when squared each element gets
    multiplied by its weight, µ(g).
    """
    b0   = np.row_stack((np.column_stack((b, np.zeros(b.shape))),
                         np.column_stack((np.zeros(b.shape), b))))
    Ymin = np.minimum(Y - X.dot(b0), np.zeros(Y.shape))
    return (1e3 * Ymin * np.kron(np.ones((1, 2)), mu_g ** 0.5)).flatten()



def estimateBoundsMC(x, xi, rc_seed, share, mk_id, ns, sigma, beta0, iota = 1e-10, alt = False):
    """Estimate demand model using a bounds approach; see estimateBounds for details."""

    # Bounds approach
    # ---------------

    M = len(np.unique(mk_id))
    J = int(np.unique(np.diff(np.where(np.diff(mk_id)))))

    eta  = (1 - iota) / (ns + J + 1)
    r_n  = 50  # the r runs from 1 to r_n

    mu_g, g1 = g_func(x, r_n)
    g = np.column_stack((g1, np.ones((g1.shape[0], 1))))

    # weights on the constant is equal to some constant bigger than the weight of
    # the coarsest hypercube of the x
    mu_g[g.shape[1] - 1] = 1 / 202

    # Starting value of ξ_{jt}
    # ------------------------

    x1    = np.column_stack((np.ones((x.shape[0], 1)), x))
    delta = x1.dot(beta0).reshape((-1, 1)) + xi
    exp_delta_bound = np.exp(delta)

    # Laplace shares
    # --------------

    dummies    = np.row_stack((mk_id == t for t in range(M))).astype(int)
    share_lap  = (share * ns + 1) / (ns + J + 1)
    out_ch_aux = ns - dummies.dot(share * ns)
    out_ch     = np.array([t for j in range(J) for t in out_ch_aux])
    share_lap0 = (out_ch + 1) / (ns + J + 1)

    # Actual optimization
    # -------------------

    argsBounds = (rc_seed,
                  x,
                  exp_delta_bound,
                  share_lap,
                  share_lap0,
                  mk_id,
                  M,
                  eta,
                  g,
                  mu_g,
                  beta0)

    kwargsOptim = {"xtol": 1e-7, "ftol": 1e-7, "disp": False}

    global beta
    beta = np.empty(beta.shape)
    if alt:
        objectiveBoundsMC(sigma, *argsBounds)
        sigma_bounds = sigma
    else:
        sigma_bounds = fmin(func = objectiveBoundsMC,
                            args = argsBounds,
                            x0   = sigma + 0.1,
                            **kwargsOptim)
    return sigma_bounds, beta


def objectiveBoundsMC(sigma,
                      rc_seed,
                      x,
                      exp_delta_bound,
                      share_lap,
                      share_lap0,
                      mk_id,
                      M,
                      eta,
                      g,
                      mu_g,
                      beta0):
    """Bounds objective (min_Σ) min_β Q_T(β; Σ); MC version.

    See objectiveBounds for details."""

    # exp(x_{jt} Σ_0 v_i)
    rc_sd     = sigma * rc_seed
    mu_x      = rc_sd * x
    x_ind_exp = np.exp(mu_x)

    # Estimate δ_{jt} using iterative prodedure, similar to BLP
    delta_lap = meanval(exp_delta_bound, share_lap, x_ind_exp, mk_id, M)

    # Compute bounds for δ_{jt} following Gandhi et al. (2017)
    share_ratio = share_lap0 / share_lap
    delta_u     = delta_lap + np.log(share_ratio * ((share_lap + eta) / (share_lap0 - eta)))
    delta_l     = delta_lap + np.log(share_ratio * ((share_lap - eta) / (share_lap0 + eta)))

    # Compute the elements of ρ^l_T(θ, g) and ρ^u_T(θ, g)
    y_u   = delta_u
    y_l   = -delta_l
    X     = np.column_stack((np.ones(x.shape), x))
    Y_sum = g.T.dot(np.column_stack((y_l, y_u)))  # a, b
    X_sum = g.T.dot(np.column_stack((-X, X)))     # c, d
    Maux  = 1 / M ** 0.5

    # Return the objective (minimize Q wrt β; see estimateQ for more)
    return estimateQ(Maux * Y_sum, Maux * X_sum, beta0, mu_g)
