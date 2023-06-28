# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

# -----------------------------------------------------------------------------
# Main wrapper function
# -----------------------------------------------------------------------------

def squarem(func, x0, args = (), fmerit = None,
            method = 3, K = 1, **kwargs):
    """Wrapper for SQUAREM acceleration of any fixed-point iteration

    This was adapted to python from Ravi Varadhan's R package, SQUAREM. See
    Varadhan and Roland (2008) and Varadhan (2016) for details. Much of the
    function and documentation is taken verbatim.

    Args
    ----

    func : callable
        Function F that denotes the fixed-point mapping to run during the
        iteration. x0 must be a real-valued vector of length N and func must
        be F: R^n -> R^n. We loop for x^{k + 1} = F(x^k)
    x0 : numpy.ndarray
        starting value for the arguments of F

    Kwargs
    ------

    args : tuple
        Arguments to pass to func
    fmerit : callable
        This is a scalar function L: R^n -> R that denotes a ”merit”
        function which attains its local minimum at the fixed-point of
        F. In the EM algorithm, the merit function L is the negative of
        log-likelihood. In some problems, a natural merit function may not
        exist, in which case the algorithm works with only F. The merit
        function function does not have to be specified, even when a natural
        merit function is available, and especially when its computation is
        expensive.
    method : int
        Stepping method; must be 1, 2, or 3 (default). These correspond to the
        3 schemes discussed in Varadhan and Roland (2008).
            1) alpha_k = r ⋅ v / v ⋅ v
            2) alpha_k = r ⋅ r / r ⋅ v
            3) alpha_k = - sqrt(r ⋅ r / v ⋅ v)
        where r = F(x) - x, v = F(F(x)) - 2F(x) + x
    K : int (not yet implemented)
        Order of minimal-polynomial extrapolation. Varadhan (2016) suggests
        using first-order schemes, and notes that K = 2, 3 may provide greater
        speed in some instances, but are less reliable than first-order
        schemes. We have not yet implemented K > 1.
    square : bool (not yet implemented)
        Whether to use "squaring" with the extrapolation. Varadhan (2016)
        notes this typically makes the iteration faster.
    step_min : float
        Minimum step length. 1 works well for contractive fixed-point
        iterations (e.g. EM and MM). In problems where an eigenvalue of the
        Jacobian of the F is outside of the interval (0, 1), this should be
        less than 1 or even negative in some cases.
    step_max : float
        The startng vaue of the maximum step length. When the step length
        exceeds step_max, it is set equal to step_max, but then step_max is
        increased by a factor of mstep.
    mstep : float
        Scalar > 1. When the step length exceeds step_max, it is set to
        step_max but then step_max is in turn set to mstep * step_max.
    fmerit_inc : float
        Non-negative scalar that dictates the degree of non-monotonicity.
        Defaults to 1; set it to 0 to obtain monotone convergence. Setting
        fmerit_inc to np.inf gives a non-monotone scheme. In-between values
        result in partially-monotone convergence.
    kr : float
        Non-negative scalar that dictates the degree of non-monotonicity.
        Defaults to 1; set it to 0 to obtain monotone convergence. Setting
        fmerit_inc to np.inf gives a non-monotone scheme. In-between values
        result in partially-monotone convergence. This parameter is only used
        when fmerit is not specified by user.
    xtol : float
        Tolerance for ||x^{k + 1} - x^k||
    maxiter : int
        Maximum number of function evaluations.
    lnorm : callable
        Custom norm to use for comparing x^{k + 1}, x^k. Defaults to None to
        use the L2 norm (euclidean distance).
    disp : bool
        Whether to trace the iterations.
    disp_every : int
        Display iteration info every disp_every iterations, otherwise an
        iteration is only marked by a dot, "."

    Returns
    -------

    ResultsSQUAREM object with the following attributes:

    x : ndarray
        The solution of the fixed point iteration
    fmeritval : float (only if fmerit was specified)
        Value of the merit function
    success : bool
        Whether or not the iteration exited successfully.
    message : str
        Description of the cause of the termination.
    feval, fmeriteval (only if fmerit was specified), niter : int
        Number of evaluations of the function, the merit functions, and
        the number of iterations performed.

    References
    ----------

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.

    - Ravi Varadhan (2016). SQUAREM: Squared Extrapolation Methods for
      Accelerating EM-Like Monotone Algorithms. R package version 2016.8-2.
      https://CRAN.R-project.org/package=SQUAREM
    """

    if method not in [1, 2, 3]:
        raise Warning("method must be one of [1, 2, 3]")

    if K != 1:
        raise Warning("K != 1 not yet implemented.")

    # if (K > 1) and (method not in ["rre", "mpe"]):
    #     method = "rre"
    #
    # if (K == 1) and (method not in [1, 2, 3]):
    #     method = 3

    kwargsDefault = {
        "method":     method,
        "K":          K,
        "square":     True,
        "step_min":   1,
        "step_max":   1,
        "mstep":      4,
        "kr":         1,
        "disp":       False,
        "disp_every": 10,
        "fmerit_inc": 1,
        "xtol":       1e-7,
        "maxiter":    1500,
        "lnorm":      None
    }

    for key in kwargs.keys():
        if key not in kwargsDefault.keys():
            msg = "squarem got an unexpected keyword argument '{0}'"
            raise TypeError(msg.format(key))

    kwargsDefault.update(kwargs)

    if fmerit is not None:
        if (K == 1):
            results = squarem1(func, x0, fmerit, args, **kwargsDefault)
        elif (K > 1) or (method in ["rre", "mpe"]):
            results = cyclem1(func, x0, fmerit, args **kwargsDefault)
    else:
        if (K == 1):
            results = squarem2(func, x0, args, **kwargsDefault)
        elif (K > 1) or (method in ["rre", "mpe"]):
            results = cyclem2(func, x0, args, **kwargsDefault)

    return results


# -----------------------------------------------------------------------------
# Partially monotone, globally-convergent acceleration
# -----------------------------------------------------------------------------

def squarem1(func,
             x0,
             fmerit,
             args = (),
             method = 3,
             step_min = 1,
             step_max = 1,
             mstep = 4,
             fmerit_inc = 1,
             xtol = 1e-7,
             maxiter = 1500,
             lnorm = None,
             disp = False,
             disp_every = 10,
             K = 1,
             square = True,
             kr = 1):
    """Partially-monotone, globally-convergent SQUAREM acceleration

    Can accelerate EM, MM, and other fixed-point iterations. Here an adaptive
    maximum step-size strategy is used. This strategy is effective and
    represents a good trade-off between speed and stability.

    This was adapted to python from Ravi Varadhan's R package, SQUAREM. See
    Varadhan and Roland (2008) and Varadhan (2016) for details. Much of the
    function and documentation is taken verbatim.

    Args
    ----

    func : callable
        Function F that denotes the fixed-point mapping to run during the
        iteration. x0 must be a real-valued vector of length N and func must
        be F: R^n -> R^n. We loop for x^{k + 1} = F(x^k)
    x0 : numpy.ndarray
        starting value for the arguments of F
    fmerit : callable
        This is a scalar function L: R^n -> R that denotes a ”merit”
        function which attains its local minimum at the fixed-point of
        F. In the EM algorithm, the merit function L is the negative of
        log-likelihood. In some problems, a natural merit function may not
        exist, in which case the algorithm works with only F. The merit
        function function does not have to be specified, even when a natural
        merit function is available, and especially when its computation is
        expensive.

    Kwargs
    ------

    args : tuple
        Arguments to pass to func
    method : int
        Stepping method; must be 1, 2, or 3 (default). These correspond to the
        3 schemes discussed in Varadhan and Roland (2008).
            1) alpha_k = r ⋅ v / v ⋅ v
            2) alpha_k = r ⋅ r / r ⋅ v
            3) alpha_k = - sqrt(r ⋅ r / v ⋅ v)
        where r = F(x) - x, v = F(F(x)) - 2F(x) + x
    step_min : float
        Minimum step length. 1 works well for contractive fixed-point
        iterations (e.g. EM and MM). In problems where an eigenvalue of the
        Jacobian of the F is outside of the interval (0, 1), this should be
        less than 1 or even negative in some cases.
    step_max : float
        The startng vaue of the maximum step length. When the step length
        exceeds step_max, it is set equal to step_max, but then step_max is
        increased by a factor of mstep.
    mstep : float
        Scalar > 1. When the step length exceeds step_max, it is set to
        step_max but then step_max is in turn set to mstep * step_max.
    fmerit_inc : float
        Non-negative scalar that dictates the degree of non-monotonicity.
        Defaults to 1; set it to 0 to obtain monotone convergence. Setting
        fmerit_inc to np.inf gives a non-monotone scheme. In-between values
        result in partially-monotone convergence.
    xtol : float
        Tolerance for ||x^{k + 1} - x^k||
    maxiter : int
        Maximum number of function evaluations.
    lnorm : callable
        Custom norm to use for comparing x^{k + 1}, x^k. Defaults to None to
        use the L2 norm (euclidean distance).
    disp : bool
        Whether to trace the iterations.
    disp_every : int
        Display iteration info every disp_every iterations, otherwise an
        iteration is only marked by a dot, "."; defaults to 10.
    K : int (unused)
        Argument passed to normalize squarem wrapper call.
    square : bool (unused)
        Argument passed to normalize squarem wrapper call.
    kr : float (unused)
        Argument passed to normalize squarem wrapper call.

    Returns
    -------

    ResultsSQUAREM object with the following attributes:

    x : ndarray
        The solution of the fixed point iteration
    fmeritval : float
        Value of the merit function
    success : bool
        Whether or not the iteration exited successfully.
    message : str
        Description of the cause of the termination.
    feval, fmeriteval, niter : int
        Number of evaluations of the function, the merit functions, and
        the number of iterations performed.

    References
    ----------

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.

    - Ravi Varadhan (2016). SQUAREM: Squared Extrapolation Methods for
      Accelerating EM-Like Monotone Algorithms. R package version 2016.8-2.
      https://CRAN.R-project.org/package=SQUAREM
    """

    if mstep < 1:
        raise Warning("mstep should be > 1.")

    msg = r"iter {0:,}: ||Δx|| = {1:.6g}, obj = {2:,.6g}, extrap = {3}, step = {4:.3g}"
    l2norm = lnorm is None
    step_max0 = step_max
    # if disp:
    #     print("SQUAREM-1")

    i     = 1
    lold  = fmerit(p, *args)
    leval = 1
    feval = 0

    if method == 1:
        step_fun = step1
    elif method == 2:
        step_fun = step2
    elif method == 3:
        step_fun = step3

    while (feval < maxiter):
        extrap = True
        x1     = func(x0, *args)
        feval += 1

        q1   = x1 - x0
        sr2  = crossprod(q1)
        norm = sr2 ** 0.5 if l2norm else lnorm(x1, x0)
        if (norm < xtol):
            break

        x2     = func(x1, *args)
        feval += 1

        q2   = x2 - x1
        sq2  = np.sqrt(crossprod(q2))
        norm = sq2 if l2norm else lnorm(x2, x1)
        if (sq2 < xtol):
            break

        sv2 = crossprod(q2 - q1)
        srv = np.vdot(q1, q2 - q1)

        alpha = max(step_min, min(step_max, step_fun(sr2, sv2, srv)))
        x_new = x0 + 2 * alpha * q1 + alpha ** 2 * (q2 - q1)
        error = False

        if (abs(alpha - 1) > 0.01):
            try:
                x_new = func(x_new, *args)
            except:
                error = True
            finally:
                feval += 1

        if (error or np.any(np.isnan(x_new))):
            x_new  = x2
            lnew   = fmerit(x2, *args)
            leval += 1
            if (alpha == step_max):
                step_max = max(step_max0, step_max / mstep)

            alpha  = 1
            extrap = False
        else:
            error = False
            if np.isfinite(fmerit_inc):
                try:
                    lnew = fmerit(x_new, *args)
                except:
                    error = True
                finally:
                    leval += 1
            else:
                lnew = lold

            if error or np.isnan(lnew) or (lnew > (lold + fmerit_inc)):
                x_new  = x2
                lnew   = fmerit(x2, *args)
                leval += 1
                if (alpha == step_max):
                    step_max = max(step_max0, step_max / mstep)

                alpha  = 1
                extrap = False

        if (alpha == step_max):
            step_max = mstep * step_max

        if (step_min < 0) and (alpha == step_min):
            step_min = mstep * step_min

        x0 = x_new
        if not np.isnan(lnew):
            lold = lnew

        if disp:
            if (i % disp_every):
                print(".", end = "")
            else:
                print("\t" + msg.format(i, norm, lnew, extrap, alpha))

        i += 1

    # if disp and (i > disp_every) and ((i - 1) % disp_every):
    if disp and ((i - 1) % disp_every):
        print((disp_every - ((i - 1) % disp_every)) * " ", end = "")
        print("\t" + msg.format(i, norm, lnew, extrap, alpha))

    if np.isinf(fmerit_inc):
        lold   = fmerit(x0, *args)
        leval += 1

    success = (feval < maxiter)
    if success:
        message = msg.format(i, norm, lnew, extrap, alpha)
    else:
        message = "Too many function evaluations required"

    results            = ResultsSQUAREM
    results.x          = x0
    results.i          = i
    results.feval      = feval
    results.fmeriteval = leval
    results.success    = success
    results.fmeritval  = lold
    results.message    = message

    return results


def squarem2(func,
             x0,
             args = (),
             method = 3,
             step_min = 1,
             step_max = 1,
             mstep = 4,
             xtol = 1e-7,
             maxiter = 1500,
             lnorm = None,
             disp = False,
             disp_every = 10,
             fmerit_inc = 1,
             K = 1,
             square = True,
             kr = 1):
    """Partially-monotone, globally-convergent SQUAREM acceleration

    Can accelerate EM, MM, and other fixed-point iterations. Here an adaptive
    maximum step-size strategy is used. This strategy is effective and
    represents a good trade-off between speed and stability.

    This was adapted to python from Ravi Varadhan's R package, SQUAREM. See
    Varadhan and Roland (2008) and Varadhan (2016) for details. Much of the
    function and documentation is taken verbatim.

    Args
    ----

    func : callable
        Function F that denotes the fixed-point mapping to run during the
        iteration. x0 must be a real-valued vector of length N and func must
        be F: R^n -> R^n. We loop for x^{k + 1} = F(x^k)
    x0 : numpy.ndarray
        starting value for the arguments of F

    Kwargs
    ------

    args : tuple
        Arguments to pass to func
    method : int
        Stepping method; must be 1, 2, or 3 (default). These correspond to the
        3 schemes discussed in Varadhan and Roland (2008).
            1) alpha_k = r ⋅ v / v ⋅ v
            2) alpha_k = r ⋅ r / r ⋅ v
            3) alpha_k = - sqrt(r ⋅ r / v ⋅ v)
        where r = F(x) - x, v = F(F(x)) - 2F(x) + x
    step_min : float
        Minimum step length. 1 works well for contractive fixed-point
        iterations (e.g. EM and MM). In problems where an eigenvalue of the
        Jacobian of the F is outside of the interval (0, 1), this should be
        less than 1 or even negative in some cases.
    step_max : float
        The startng vaue of the maximum step length. When the step length
        exceeds step_max, it is set equal to step_max, but then step_max is
        increased by a factor of mstep.
    mstep : float
        Scalar > 1. When the step length exceeds step_max, it is set to
        step_max but then step_max is in turn set to mstep * step_max.
    xtol : float
        Tolerance for ||x^{k + 1} - x^k||
    maxiter : int
        Maximum number of function evaluations.
    lnorm : callable
        Custom norm to use for comparing x^{k + 1}, x^k. Defaults to None to
        use the L2 norm (euclidean distance).
    disp : bool
        Whether to trace the iterations.
    disp_every : int
        Display iteration info every disp_every iterations, otherwise an
        iteration is only marked by a dot, "."; defaults to 10.
    fmerit_inc : float (unused)
        Argument passed to normalize squarem wrapper call.
    K : int (unused)
        Argument passed to normalize squarem wrapper call.
    square : bool (unused)
        Argument passed to normalize squarem wrapper call.
    kr : float (unused)
        Argument passed to normalize squarem wrapper call.

    Returns
    -------

    ResultsSQUAREM object with the following attributes:

    x : ndarray
        The solution of the fixed point iteration
    success : bool
        Whether or not the iteration exited successfully.
    message : str
        Description of the cause of the termination.
    feval, niter : int
        Number of evaluations of the function, and the number of iterations
        performed.

    References
    ----------

    - Varadhan, Ravi and Roland, Christophe. (2008). Simple and Globally
      Convergent Methods for Accelerating the Convergence of Any EM Algorithm.
      Scandinavian Journal of Statistics, 35(2): 335–353.

    - Ravi Varadhan (2016). SQUAREM: Squared Extrapolation Methods for
      Accelerating EM-Like Monotone Algorithms. R package version 2016.8-2.
      https://CRAN.R-project.org/package=SQUAREM
    """

    if mstep < 1:
        raise Warning("mstep should be > 1.")

    msg = r"iter {0:,}: ||Δx|| = {1:.6g}, extrap = {2}, step = {3:.3g}"
    l2norm = lnorm is None
    step_max0 = step_max

    i     = 1
    feval = 0
    kount = 0

    if method == 1:
        step_fun = step1
    elif method == 2:
        step_fun = step2
    elif method == 3:
        step_fun = step3

    # if disp:
    #     print("SQUAREM-2")

    while (feval < maxiter):
        extrap = True
        x1     = func(x0, *args)
        feval += 1

        q1   = x1 - x0
        sr2  = crossprod(q1)
        norm = sr2 ** 0.5 if l2norm else lnorm(x1, x0)
        if (norm < xtol):
            break

        x2     = func(x1, *args)
        feval += 1

        q2   = x2 - x1
        sq2  = res = np.sqrt(crossprod(q2))
        norm = res if l2norm else lnorm(x2, x1)
        if (norm < xtol):
            break

        sv2 = crossprod(q2 - q1)
        srv = np.vdot(q1, q2 - q1)

        alpha = max(step_min, min(step_max, step_fun(sr2, sv2, srv)))
        x_new = x0 + 2 * alpha * q1 + alpha ** 2 * (q2 - q1)

        if (abs(alpha - 1) > 0.01):
            try:
                x_tmp = func(x_new, *args)
                assert not np.any(np.isnan(x_tmp))

                res     = crossprod(x_tmp - x_new) ** 0.5
                parnorm = (crossprod(x2) / x2.shape[0]) ** 0.5
                kres    = kr * (1 + parnorm) + sq2
                x_new   = x_tmp if (res <= kres) else p2
                if (res > kres):
                    if (alpha == step.max):
                        step_max = max(step_max0, step_max / mstep)

                    alpha  = 1
                    extrap = False
            except:
                x_new = x2
                if (alpha == step_max):
                    step_max = max(step_max0, step_max / mstep)

                alpha  = 1
                extrap = False
            finally:
                feval += 1

        if (alpha == step_max):
            step_max = mstep * step_max

        if (step_min < 0) and (alpha == step_min):
            step_min = mstep * step_min

        if disp:
            if (i % disp_every):
                print(".", end = "")
            else:
                print("\t" + msg.format(i, norm, extrap, alpha))

        x0 = x_new
        i += 1

    # if disp and (i > disp_every) and ((i - 1) % disp_every):
    if disp and ((i - 1) % disp_every):
        print((disp_every - ((i - 1) % disp_every)) * " ", end = "")
        print("\t" + msg.format(i, norm, extrap, alpha))

    success = (feval < maxiter)
    if success:
        message = msg.format(i, norm, extrap, alpha)
    else:
        message = "Too many function evaluations required"

    results          = ResultsSQUAREM
    results.x        = x0
    results.i        = i
    results.feval    = feval
    results.success  = success
    results.message  = message

    return results

# -----------------------------------------------------------------------------
# Fixed point iteration
# -----------------------------------------------------------------------------

def fixed_point(func, x0,
                args = (),
                xtol = 1e-7,
                maxiter = 5000,
                lnorm = None,
                disp = False,
                disp_every = 50):
    """Basic iterative scheme to find the fixed point of a function F

    Args
    ----
    
    func : callable
        Function F that denotes the fixed-point mapping to run during the
        iteration. x0 must be a real-valued vector of length N and func must
        be F: R^n -> R^n. We loop for x^{k + 1} = F(x^k)
    x0 : numpy.ndarray
        starting value for the arguments of F

    Kwargs
    ------

    args : tuple
        Arguments to pass to func
    xtol : float
        Tolerance for ||x^{k + 1} - x^k||
    maxiter : int
        Maximum number of function evaluations.
    lnorm : callable
        Custom norm to use for comparing x^{k + 1}, x^k. Defaults to None to
        use the L2 norm (euclidean distance).
    disp : bool
        Whether to trace the iterations.
    disp_every : int
        Display iteration info every disp_every iterations, otherwise an
        iteration is only marked by a dot, "."; defaults to 50.

    Returns
    -------

    ResultsSQUAREM object with the following attributes:

    x : ndarray
        The solution of the fixed point iteration
    success : bool
        Whether or not the iteration exited successfully.
    message : str
        Description of the cause of the termination.
    niter : int
        Number of evaluations of the function.
    """

    msg = "iter {0:,}: ||Δx|| = {1:.6g}"

    i = 1
    l2norm = lnorm is None
    success = False
    while (i < maxiter):
        x1 = func(x0, *args)
        norm = np.sqrt(crossprod(x1 - x0)) if l2norm else lnorm(x1, x0)
        if (norm < xtol):
            success = True
            break

        if disp:
            if (i % disp_every):
                print(".", end = "")
            else:
                print("\t" + msg.format(i, norm))

        x0 = x1
        i += 1

    results   = ResultsSQUAREM
    results.x = x0
    results.i = i

    # if disp and (i > disp_every) and ((i - 1) % disp_every):
    if disp and ((i - 1) % disp_every):
        print((disp_every - ((i - 1) % disp_every)) * " ", end = "")
        print("\t" + msg.format(i, norm))

    if success:
        message = msg.format(i, norm)
    else:
        message = "Too many function evaluations required"

    results.success  = success
    results.message  = message

    return results


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def crossprod(x):
    return (x ** 2).sum()


def step1(sr2, sv2, srv):
    return -srv / sv2


def step2(sr2, sv2, srv):
    return -sr2 / srv


def step3(sr2, sv2, srv):
    return np.sqrt(sr2 / sv2)


class ResultsSQUAREM(dict):
    """Represents the result from SQUAREM

    Attributes
    ----------

    x : ndarray
        The solution of the fixed point iteration
    fmerit_value : float (optional)
        Value of the fmerit function
    success : bool
        Whether or not the iteration exited successfully.
    message : str
        Description of the cause of the termination.
    feval, fmeriteval (optional), niter : int
        Number of evaluations of the function, the merit functions, and
        the number of iterations performed.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
