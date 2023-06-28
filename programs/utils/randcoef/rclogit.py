# -*- coding: utf-8 -*-

from __future__ import division, print_function
from time import time
import pandas as pd
import numpy as np

# from utils.randcoef.blp import estimateBLP
# from utils.randcoef.bounds import estimateBounds
# from utils.randcoef.common import encodeVariables, getDummies
from .blp import estimateBLP
from .bounds import estimateBounds
from .common import encodeVariables, getDummies


def rclogit(df,
            shares,
            xLinear,
            xNonLinear,
            instruments,
            market,
            dfMarketSizes,
            marketSizes,
            R,
            addConstant = True,
            fixedEffects = None,
            dfDemographics = None,
            demographics = None,
            shareTotals = True,
            sharesTransform = None,
            estimator = "bounds",
            bootstrap = 0,
            **kwargs):
    """Random-coefficients logit model

    This estimates β_0, diag(Σ_0) in the model

        u_{ijt} = δ_{ij} + ϵ_{ijt}
                = x_jt β_0 + x_{jt} Σ_0 v_i + x_{jt} Π_0 D_i + ξ_{jt} + ϵ_{ijt}

    using the bounds estimator proposed by Gandhi et al. (2017) or the BLP
    estimator proposed by BLP (1995).

    Args
    ----

    df : pandas.DataFrame
        market-product data frame containing all the relevant variables
    shares : str
        s_{jt}, product market shares
    xLinear : list of strings
        x^1_{jt}, product chars that enter utility linearly
    xNonLinear : list of strings
        x^2_{jt}, product chars that enter utility non-linearly
    instruments : list of strings
        z_{jt}, instruments for xLin. You MUST specify the exogenous
        covariates here as well.
    market : list of strings
        Columns that specify a market
    dfMarketSizes : pandas.DataFrame
        market-level data frame containing market sizes
    marketSizes : str
        ns_t, column with market sizes in dfMarketSizes
    dfDemographics : pandas.DataFrame
        consumer-level demographics
    demographics : str
        columns with consumer demographics
    R : int
        Number of samples for the random coefficients for xNonLinear

    Kwargs
    ------

    addConstant : bool
        Add a constant to the linear parameters
    fixedEffects : list of strings
        columns that specify fixed effects; the within estimator is computed
        in case this is specified, that is, the model is multiplied by M_FE,
        I - FE (FE' FE)^-1 FE, which de-means all other variables.
    shareTotals : bool
        Whether the shares variable contains product market shares or
        product market totals. Defaults to True for the variable denoting
        s_{jt} in [0, 1]. If set to false, the latter is assumed (i.e.
        ns_t * s_{jt}).
    sharesTransform : str
        Transformation to apply to the product shares. Defaults to None
        for no transformation. At the moment only "laplace" can be specified
        for the laplace transform, (s_{jt} * ns_t + 1) / (ns_t + J_t + 1),
        where J_t is the number of products per market.
    estimator : str
        - "bounds": Gandhi et al. (2017) bounds estimator
        - "BLP": BLP (1995) estimator

    kwargs is passed to estimateBLP or estimateBounds.

    References
    ----------

    - Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices In
      Market Equilibrium. Econometrica, 63(4), 841.

    - Nevo, A. (2000). A Practitioner’s Guide to Estimation of
      Random-Coefficients Logit Models of Demand. Journal of Economics &
      Management Strategy, 9(4), 513–548.

    - Gandhi, Amit, Zhentong Lu, and Xiaoxia Shi (2017). Estimating Demand for
      Differentiated Products with Zeroes in Market Share Data. Working Paper.
    """

    # Set up, parsing, etc.
    # ---------------------

    if sharesTransform is not None:
        if sharesTransform != "laplace":
            raise Warning("Don't know sharesTransform '{0}".format(sharesTransform))

    if estimator not in ["bounds", "BLP"]:
        raise Warning("Don't know estimator '{0}".format(estimator))

    if estimator == "BLP":
        estimate = estimateBLP
    else:
        estimate = estimateBounds
        if sharesTransform is not None:
            raise Warning("sharesTransform only allowed with estimator 'BLP'")

    xNL    = unique(xNonLinear)
    dfVars = unique([shares] + xLinear + xNL + instruments)
    if fixedEffects is not None:
        feVars = unique(fixedEffects)
        for fe in feVars:
            if fe in dfVars:
                msg = "Fixed effect '{0}' cannot be anywhere else in the model"
                raise Warning(msg.format(fe))

        dfVars += feVars
    else:
        feVars = None

    # Grab only relevant variables and map market variables to singe ID
    # -----------------------------------------------------------------

    dfSubset = df[market + dfVars].set_index(market)
    dfSubset.sort_index(inplace = True)
    Jt_gen   = (group.shape[0] for name, group in dfSubset.groupby(level = market))
    Jt       = np.fromiter(Jt_gen, int)
    M        = len(Jt)
    N        = dfSubset.shape[0]
    mkID     = np.repeat(np.arange(M), Jt)
    dfSubset["mkID"] = mkID

    # Same mapping for data set with market sizes
    # -------------------------------------------

    dfMarketSubset = dfMarketSizes[market + [marketSizes]].set_index(market).sort_index()
    if dfMarketSubset.shape[0] != M:
        raise Warning("Number of markets not consistent across data sets")
    else:
        ixMarketCheck = dfSubset.index.drop_duplicates()
        if not np.all(ixMarketCheck == dfMarketSubset.index):
            raise Warning("Market levels not equal across data sets.")

    dfMarketSubset["mkID"] = np.arange(M)

    # Transform to shares or to laplace shares if applicable
    # ------------------------------------------------------

    ns_t = dfMarketSubset[marketSizes].values
    if not shareTotals:
        if sharesTransform is None:
            dfSubset[shares] = dfSubset[shares] / ns_t[dfSubset["mkID"]]
        elif sharesTransform == "laplace":
            dfSubset[shares] = (dfSubset[shares] + 1) / (ns_t + Jt + 1)[dfSubset["mkID"]]
    elif sharesTransform == "laplace":
        dfSubset[shares] = (dfSubset[shares] * ns_t[dfSubset["mkID"]] + 1) / (ns_t + Jt + 1)[dfSubset["mkID"]]

    # Add constant if requested
    # -------------------------

    # TODO: Automagically not do this with fixedEffects? // 2017-08-22 12:29 EDT
    if addConstant:
        consLabel = "cons"
        while consLabel in xLinear:
            consLabel = "_" + consLabel

        xLinearCons = unique(xLinear + [consLabel])
        dfSubset[consLabel] = np.ones(N)
    else:
        xLinearCons = unique(xLinear)

    # FE transform (de-mean if applicable)
    # ------------------------------------

    if estimator == "BLP":
        dfSubset = dfSubset.query("{0} > 0".format(shares))

    zTransform = unique(instruments)
    if fixedEffects is not None:
        if estimator == "bounds":
            print("WARNING: Fixed effects can slow down the computation.")
            feMatrix    = getDummies(dfSubset, feVars)
            xDemeaned   = np.column_stack((dfSubset[xLinearCons].values, feMatrix))
            zDemeaned   = np.column_stack((dfSubset[zTransform].values,  feMatrix))
            feIndex     = None
            feTransform = None
        else:
            dfTransform  = dfSubset.groupby(feVars)
            xDemeaned    = (dfSubset[xLinearCons] - dfTransform[xLinearCons].transform('mean')).values
            zDemeaned    = (dfSubset[zTransform] - dfTransform[zTransform].transform('mean')).values
            feIndex      = encodeVariables(dfSubset, cols = feVars)
            feTransform  = pd.get_dummies(feIndex).values
            feTransform /= feTransform.sum(0)
            feTransform  = feTransform.T
    else:
        xDemeaned   = None
        zDemeaned   = None
        feIndex     = None
        feTransform = None

        # TODO: Though this is fine in theory, in practice there's no way
        # your PC will be able to invert an n by n matrix for large n. Even
        # at 10M observations that's already ~100GiB. // 2017-08-21 18:27 EDT
        # dummies = getDummies(dfSubset, feVars)
        # within  = np.identity(n) - dummies.dot(np.linalg.pinv(dummies.T.dot(dummies)).dot(dummies.T))

    # Run the actual optimization
    # ---------------------------

    results = RCLogitResults()

    results.sharesTransform = sharesTransform
    results.instruments     = zTransform
    results.absorbed        = feVars
    results.estimator       = estimator
    results.sigmaVars       = xNL
    results.betaVars        = xLinearCons
    if xDemeaned is not None:
        results.betaVars += ["_fe{0}" for f in range(len(feVars))]

    if dfDemographics is not None:

        # If demographics are provided, estimate their coefficients as well
        # -----------------------------------------------------------------

        if demographics is None:
            colsDemo = unique(dfDemographics.columns)
        else:
            colsDemo = unique(demographics)

        results.piVars    = colsDemo
        results.thetaVars = results.sigmaVars + results.piVars + results.betaVars

        # TODO: Check R and Di.shape are consistent // 2017-08-21 12:57 EDT
        sigma, pi, beta = estimate(dfSubset[[shares]].values,
                                   dfSubset[xLinearCons].values if xDemeaned is None else xDemeaned,
                                   dfSubset[xNL].values,
                                   dfSubset[zTransform].values if zDemeaned is None else zDemeaned,
                                   dfSubset["mkID"].values,
                                   ns_t.reshape((-1, 1)),
                                   R,
                                   Di = dfDemographics[colsDemo].values.T,
                                   feIndex = feIndex,
                                   feTransform = feTransform,
                                   **kwargs)

        results.sigma = sigma
        results.beta  = beta
        results.pi    = pi
        results.theta = np.concatenate((sigma.flatten(), pi.flatten(), beta.flatten()))

        # Bootstrap standard errors, if requested
        # ---------------------------------------

        if bootstrap > 0:
            bootindex    = np.arange(dfSubset.shape[0])
            mkIDx, mkIDy = dfSubset["mkID"].values, dfMarketSubset["mkID"].values
            bootselect   = np.array([bootindex[mkIDx == t] for t in mkIDy])
            bootlens     = np.array([len(s) for s in bootselect])
            Bresults     = np.empty((0, len(sigma) + len(beta)))

            timerBase  = time()
            timerMsg   = "\tbootstrap iter #{0}: {1:,.2f} minutes"
            for b in range(bootstrap):
                bootsample = np.random.randint(M, size = M)
                ns_t       = dfMarketSubset.iloc[bootsample][[marketSizes]].values
                dfBoot     = dfSubset.iloc[np.concatenate(bootselect[bootsample])]
                mkID       = np.repeat(np.arange(M), bootlens[bootsample])

                if fixedEffects is not None:
                    if estimator == "bounds":
                        feMatrix     = getDummies(dfBoot, feVars)
                        xDemeaned    = np.column_stack((dfBoot[xLinearCons].values, feMatrix))
                        zDemeaned    = np.column_stack((dfBoot[zTransform].values,  feMatrix))
                        feIndex      = None
                        feTransform  = None
                    else:
                        dfTransform  = dfBoot.groupby(feVars)
                        xDemeaned    = (dfBoot[xLinearCons] - dfTransform[xLinearCons].transform('mean')).values
                        zDemeaned    = (dfBoot[zTransform] - dfTransform[zTransform].transform('mean')).values
                        feIndex      = encodeVariables(dfBoot, cols = feVars)
                        feTransform  = pd.get_dummies(feIndex).values
                        feTransform /= feTransform.sum(0)
                        feTransform  = feTransform.T
                else:
                    xDemeaned   = None
                    zDemeaned   = None
                    feIndex     = None
                    feTransform = None

                if "debugLevel" in kwargs.keys():
                    if type(kwargs["debugLevel"]) is int:
                        kwargs["debugLevel"] = 0

                Bsigma, Bpi, Bbeta = estimate(dfBoot[[shares]].values,
                                              dfBoot[xLinearCons].values if xDemeaned is None else xDemeaned,
                                              dfBoot[xNL].values,
                                              dfBoot[zTransform].values if zDemeaned is None else zDemeaned,
                                              mkID,
                                              ns_t,
                                              R,
                                              Di = dfDemographics[colsDemo].values.T,
                                              feIndex = feIndex,
                                              feTransform = feTransform,
                                              **kwargs)

                Bresults = np.row_stack((Bresults, np.concatenate((Bsigma, Bpi.flatten(), Bbeta))))

                timerDelta = (time() - timerBase) / 60
                timerBase  = time()
                print(timerMsg.format(b, timerDelta))

            results.thetaSE = Bresults.std(0)
            results.sigmaSE = results.thetaSE[:len(sigma)]
            results.piSE    = results.thetaSE[len(sigma):-len(beta)].reshape(pi.shape)
            results.betaSE  = results.thetaSE[-len(beta):]

            cil, ciu = np.percentile(Bresults, [2.5, 97.5], axis = 0)
            results.thetaCILower = cil
            results.thetaCIUpper = ciu
            results.sigmaCILower = cil[:len(sigma)]
            results.sigmaCIUpper = ciu[:len(sigma)]
            results.piCILower    = cil[len(sigma):-len(beta)].reshape(pi.shape)
            results.piCIUpper    = ciu[len(sigma):-len(beta)].reshape(pi.shape)
            results.betaCILower  = cil[-len(beta):]
            results.betaCIUpper  = ciu[-len(beta):]

        return results
    else:

        # Otherwise only estimate the random coefficients
        # -----------------------------------------------

        results.piVars    = None
        results.thetaVars = results.sigmaVars + results.betaVars
        sigma, beta = estimate(dfSubset[[shares]].values,
                               dfSubset[xLinearCons].values if xDemeaned is None else xDemeaned,
                               dfSubset[xNL].values,
                               dfSubset[zTransform].values if zDemeaned is None else zDemeaned,
                               dfSubset["mkID"].values,
                               ns_t.reshape((-1, 1)),
                               R,
                               feIndex = feIndex,
                               feTransform = feTransform,
                               **kwargs)

        results.sigma = sigma
        results.beta  = beta
        results.pi    = None
        results.theta = np.concatenate((sigma.flatten(), beta.flatten()))

        # Bootstrap standard errors, if requested
        # ---------------------------------------

        if bootstrap > 0:
            bootindex    = np.arange(dfSubset.shape[0])
            mkIDx, mkIDy = dfSubset["mkID"].values, dfMarketSubset["mkID"].values
            bootselect   = np.array([bootindex[mkIDx == t] for t in mkIDy])
            bootlens     = np.array([len(s) for s in bootselect])
            Bresults     = np.empty((0, len(sigma) + len(beta)))

            timerBase  = time()
            timerMsg   = "\tbootstrap iter #{0}: {1:,.2f} minutes"
            for b in range(bootstrap):
                bootsample = np.random.randint(M, size = M)
                ns_t       = dfMarketSubset.iloc[bootsample][[marketSizes]].values
                dfBoot     = dfSubset.iloc[np.concatenate(bootselect[bootsample])]
                mkID       = np.repeat(np.arange(M), bootlens[bootsample])

                if fixedEffects is not None:
                    if estimator == "bounds":
                        feMatrix     = getDummies(dfBoot, feVars)
                        xDemeaned    = np.column_stack((dfBoot[xLinearCons].values, feMatrix))
                        zDemeaned    = np.column_stack((dfBoot[zTransform].values,  feMatrix))
                        feIndex      = None
                        feTransform  = None
                    else:
                        dfTransform  = dfBoot.groupby(feVars)
                        xDemeaned    = (dfBoot[xLinearCons] - dfTransform[xLinearCons].transform('mean')).values
                        zDemeaned    = (dfBoot[zTransform] - dfTransform[zTransform].transform('mean')).values
                        feIndex      = encodeVariables(dfBoot, cols = feVars)
                        feTransform  = pd.get_dummies(feIndex).values
                        feTransform /= feTransform.sum(0)
                        feTransform  = feTransform.T
                else:
                    xDemeaned   = None
                    zDemeaned   = None
                    feIndex     = None
                    feTransform = None

                if "debugLevel" in kwargs.keys():
                    if type(kwargs["debugLevel"]) is int:
                        kwargs["debugLevel"] = 0

                Bsigma, Bbeta = estimate(dfBoot[[shares]].values,
                                         dfBoot[xLinearCons].values if xDemeaned is None else xDemeaned,
                                         dfBoot[xNL].values,
                                         dfBoot[zTransform].values if zDemeaned is None else zDemeaned,
                                         mkID,
                                         ns_t,
                                         R,
                                         feIndex = feIndex,
                                         feTransform = feTransform,
                                         **kwargs)

                Bresults = np.row_stack((Bresults, np.concatenate((Bsigma, Bbeta))))

                timerDelta = (time() - timerBase) / 60
                timerBase  = time()
                print(timerMsg.format(b, timerDelta))

            results.thetaSE = Bresults.std(0)
            results.sigmaSE = results.thetaSE[:len(sigma)]
            results.betaSE  = results.thetaSE[-len(beta):]

            cil, ciu = np.percentile(Bresults, [2.5, 97.5], axis = 0)
            results.thetaCILower = cil
            results.thetaCIUpper = ciu
            results.sigmaCILower = cil[:len(sigma)]
            results.sigmaCIUpper = ciu[:len(sigma)]
            results.betaCILower  = cil[-len(beta):]
            results.betaCIUpper  = ciu[-len(beta):]

        return results


# https://stackoverflow.com/questions/480214
def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class RCLogitResults(dict):
    """Represents the result from rclogit

    Attributes
    ----------

    theta : ndarray
        The resulting coefficients, β, Σ, Π, in one array
    beta : ndarray
        Linear coefficients, β
    sigma : ndarray
        Variance for random coefficients, Σ
    pi : ndarray
        Demographic transition matrix, Π, if applicable

    thetaVars : list
        Variable names for theta
    betaVars : list
        Variable names for beta, β
    sigmaVars : list
        Variable names for sigma, Σ
    piVars : list
        Variable names for pi, Π

    instruments : list
        Name of instruments used
    absorbed : list
        Name of absorbed fixed effects, if applicable
    sharesTransform : str
        Transform of product shares, if applicable
    estimator : str
        Estimator used
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
