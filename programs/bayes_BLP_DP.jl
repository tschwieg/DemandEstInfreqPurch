using LinearAlgebra
using Distributions
using ForwardDiff
using Printf

# Notation here is: Parameter -> Hyper-parmeter in Camel-Case
# So lambdaAlpha is the alpha-prior for the lambda parameter
struct PriorBLP
    lambdaAlpha::Float64
    lambdaTheta::Float64
    betaMean::Vector{Float64}
    betaVarInv::Matrix{Float64}
    gammaDiagMu::Float64
    gammaDiagSigma::Float64
    gammaOffDiagMu::Float64
    gammaOffDiagSigma::Float64
    etaMean::Vector{Float64}
    etaVarInv::Matrix{Float64}
    ## These are DP hyper-parameters directly from Rossi (2013)
    muBar::Vector{Float64}
    aMu::Float64
    alpha::Float64
end

struct SearchParameters
    ## Variance for seach distribution for diagonal elements of the
    ## cholesky decomposition of Σ
    sigmaVarDiag::Float64
    ## Variance for seach distribution for off-diagonal elements of the
    ## cholesky decomposition of Σ
    sigmaVarOffDiag::Float64
    ## Variance for the draws of xi that determine candidate share draws
    dScale::Float64
    ## Remaining parameters deprecated and un-used.
    lambdaTVar::Float64
    lambdaDVar::Float64
    gammaVar::Float64
    alphaVar::Float64
end

mutable struct Utility
    # This is the denominator in the predict shares, faster not to have to allocate the memory
    denom::Vector{Float64}
    # This contains
    shareHolder::Vector{Float64}
    # This contains solutions to the market inversion, before we have decided to accept the draw.
    solutionFill::Matrix{Float64}
    # This contains the likelihood of the previously accepted market
    LMarket::Vector{Float64}
    # This contains the likelihood of the last inverted market, which
    # may or may not be accepted and put into LMarket.
    LMarketTemp::Vector{Float64}
    # Jac[i] is an i x i matrix containing the jacobian for markets of that size
    Jac::Vector{Matrix{Float64}}
    # sJac is a filler used for the shares in construction the
    # jacobian. Don't want to reallocate memory each time.
    sJac::Vector{Matrix{Float64}}
    # δ contains the steps in Newton's method as we invert the system.
    δ::Vector{Float64}
    # This is the δ values for the last accepted inversion. This gives
    # good initialization to Newton's method.
    warmStart::Array{Float64,2}
    # This is the logabsdet of the last accepted share draw -- saves computing two logabsdets. 
    shareJacLDet::Vector{Float64}
end

struct Data
    X::Vector{Matrix{Float64}}
    q::Array{Int64,2}
    Z::Vector{Matrix{Float64}}
    J::Vector{Int64}
    T::Int64
    searches::Vector{Int64}
    N::Int64
    K::Int64
    ## The first L of the K parameters have random coefficients
    L::Int64
    numInst::Int64
    cMap::Array{Int64,2}
end


## Takes a vector of deltas of size J and returns the inside shares 
function IndPredictShares( δ::Vector{<:Real} )
    vMax = maximum(δ)

    num = exp.(δ .- vMax)
    return num ./ (exp(-vMax) + sum(num))
end

## For a set of characteristics X and mean utilities delta,
## This averages over a distribution Γ with I consumers drawn by ζ,
## taking the individual shares for each and returning inside market shares.
function PredictSharesBLP( δ::Vector{<:Real},  X::Matrix{<:Real}, I::Int64,
                        Γ::Matrix{<:Real}, ζ::Matrix{<:Real})
    return mean( IndPredictShares( δ +  X * (Γ * ζ[i,:])) for i in 1:I)
end

## This constructs the jacobian of IndPredictShares with respect to delta.
## I.e. del s(delta), equivalent to the jacobian wrt xi.
## Takes shares, and size of the shares as inputs
function IndBuildJac( s::Vector{<:Real}, J::Int64 )
    Jac = Matrix{Real}(undef,J,J)#zeros(J,J)
    Jac .= 0.0
    for j in 1:J
        for k in 1:J
            if j == k
                Jac[j,j] = s[j]*(1-s[j])
            else
                Jac[j,k] = -s[j]*s[k]
            end
        end
    end
    return Jac
end


## This computes the derivative of PredictSharesBLP() wrt delta (xi)
function BuildXiJacBLP( X::Matrix{<:Real}, δ::Vector{<:Real}, nSumers::Int64,
                     Γ::Matrix{<:Real}, ζ::Matrix{<:Real}, J::Int64)
    return mean( IndBuildJac( IndPredictShares( δ +  X * (Γ * ζ[i,:])), J )
                  for i in 1:nSumers)
end

## This computes the derivative of PredictSharesBLP() wrt price, by
## chain ruling out the individual coefficients on price.
function BuildPriceJacBLP( X::Matrix{<:Real}, δ::Vector{<:Real}, nSumers::Int64,
                     Γ::Matrix{<:Real}, ζ::Matrix{<:Real}, J::Int64, β::Vector{Float64})
    return mean( (β[1] + (Γ * ζ[i,:])[1]) .*
                 IndBuildJac( IndPredictShares( δ +  X * (Γ * ζ[i,:])), J )
                  for i in 1:nSumers)
end


## Given a set of covariates X, shares s, distribution Γ with drawn
## consumers ζ in a market of size J, perform a BLP-inversion to
## recover xi.  Inversion begins with contraction mapping and then
## uses Werner-King to acheive desired convergence.
function FastInvertShares( X::Matrix{<:Real}, s::Vector{<:Real}, nSumers::Int64,
                           Γ::Matrix{<:Real}, ζ::Matrix{<:Real}, J::Int64,
                           delPrev::Vector{<:Real})


    
    #Since its the log jacobian we divide by PredictShares() which is exp( sHat)
    sHat = log.(PredictSharesBLP( delPrev, X, nSumers, Γ, ζ))
    sTrue = log.(s)
    diff = maximum( abs.( sHat - sTrue))

    #This doesn't need to be tight, just move us to the domain of attaction
    while( diff > 1e-1)
        delNew = delPrev + sTrue - sHat
        diff = maximum( abs.( delNew - delPrev))
        
        delPrev = delNew
        sHat = log.(PredictSharesBLP( delPrev, X, nSumers, Γ, ζ))
    end

    ## Upgrade to Werner-King
    ## Initiate the values using the contraction mapping
    its = 0
    Xn = delPrev
    Yn = delPrev

    XnNew = delPrev

    while( its < 50)
        wernerPoint = .5*Xn + .5*Yn
        sJac = BuildXiJacBLP( X, wernerPoint, nSumers, Γ, ζ, J  ) ./
            PredictSharesBLP( wernerPoint, X, nSumers, Γ, ζ)

        ## If we are outside the domain of attaction we can end up
        ## with ill-specified jacobians
        if logabsdet( sJac )[1] < -20 || any( isnan.(sJac))
            #println("Werner-King Det Problems")
            return ones(J)*NaN
        end
        
        XnNew = Xn - sJac \ (sHat - sTrue)

        sHat = log.(PredictSharesBLP( XnNew, X, nSumers, Γ, ζ))

        diff = maximum( abs.( sHat - sTrue))
        if diff <= 1e-12
            delPrev = Xn
            break
        end

        Yn = XnNew - sJac \ (sHat - sTrue)

        Xn = XnNew
        sHat = log.(PredictSharesBLP( Xn, X, nSumers, Γ, ζ))

        its += 1
    end

    if its >= 50
        return ones(J)*NaN
    end
    return delPrev
end

## MarketLikelihood. Given a set of xi, and the distribution of xi
## (exV, sd, Jac), return the likelihood of ξ
function MarketLikelihood( xi::Vector{Float64}, exV::Vector{Float64}, sd::Vector{Float64},
                           J::Int64, Jac::Matrix{Float64})

    if any( isnan.( xi))
        #println(usedContraction, " ", δ)
        return -1e20
    end
    ℓ = 0.0
    for j in 1:J
        ℓ += logpdf( Normal(exV[j], sd[j]), xi[j])
    end
    ℓ -= logabsdet( Jac )[1]
    return ℓ
end

## Given a set of data, conditional on shares s, and mapping from
## markets to arrival parameters, draw from the posterior distribution
## of lamT
function DrawLambda( data::Data, lambT::Vector{Float64},
                     newLamT::Vector{Float64}, s::Array{Float64,2},
                     pars::SearchParameters, lamMap::Vector{Vector{Int64}},
                     priors::PriorBLP)

    for (i,tCol) in enumerate( lamMap )

        postK = sum( sum( data.q[t,1:data.J[t]]) + data.searches[t] for t in tCol) +
            priors.lambdaAlpha

        postTheta = priors.lambdaTheta / ( priors.lambdaTheta*sum( 2.0 - s[t,data.J[t]+1]
                                                                   for t in tCol) + 1)
        newLamT[tCol] .= log(rand( Gamma( postK, postTheta)))
            
    end
    return newLamT
end

## Given data; arrival rates, preferences, and shock distribution, perform a
## M-H step to draw shares. 
function DrawShares( data::Data, s::Matrix{Float64}, lambT::Vector{Float64},
                     β::Vector{Float64},
                     util::Utility, newShare::Matrix{Float64},
                     pars::SearchParameters, 
                     exV::Vector{Float64}, sd::Vector{Float64},
                     cMap::Matrix{Int64}, nSumers::Int64, Γ::Matrix{Float64},
                     ζ::Matrix{Float64})

    coinFlips = log.(rand(Uniform(), data.T));

    tempShare = zeros(maximum(data.J));
    
    #The posterior likelihood of a set of shares is the poisson
    #likelihood times the prior probability coming from the inversion
    for t in 1:data.T

        if data.J[t] == 0
            continue
        end

        J = data.J[t]

        noiseAdd =  rand(Normal(0.0, pars.dScale), J)

        ## Since we know the δ, we don't need to do an inversion to get it. 
        δ = util.warmStart[t,1:J] + noiseAdd

        logdetOld = util.shareJacLDet[t]

        ## Gather the parts of the likelihood we need to do the M-H computation
        util.Jac[t] = BuildXiJacBLP( data.X[t], δ, nSumers, Γ, ζ, J )
        tempShare = PredictSharesBLP( δ, data.X[t], nSumers, Γ, ζ )

        logdetNew = logabsdet(util.Jac[t])[1]




        ## Is this a problem? I guess this makes the searches
        ## unbalanced and we need to control for this in the M-H
        ## step. This is effectively a uniform prior so we shouldn't have to though. 
        if min( minimum( tempShare[1:J]),
                1.0 - sum(tempShare[1:J])) < 1e-12
            newShare[t,1:(J+1)] = s[t,1:(J+1)]
            continue
        end

        
        λ = exp( lambT[t])

        ## First part of the likelihood is distribution of purchases conditional on shares
        qLik = 0.0
        qLikOld = 0.0
        for j in 1:J
            qLik += logpdf( Poisson( λ*tempShare[j]), data.q[t,j] )
            qLikOld += logpdf( Poisson( λ*s[t,j]), data.q[t,j] )
        end

        ## We predict shares based on δ, but likelihood uses ξ
        newLik = MarketLikelihood( δ - data.X[t]*β, exV[cMap[t,1:J]], sd[cMap[t,1:J]],
                                   J, util.Jac[t])

        if coinFlips[t] <= (newLik + qLik - util.LMarket[t] - qLikOld  +
                              logdetNew - logdetOld)

            ## Acceptinga new share means changing the stored likelihoods as well as logdetjac.
            util.warmStart[t,1:J] = δ
            util.LMarket[t] = newLik
            util.shareJacLDet[t] = logdetNew
            newShare[t,1:J] = tempShare[1:J]
            newShare[t,J+1] = 1.0 - sum( tempShare[1:J] )
        else
            newShare[t,1:(J+1)] = s[t,1:(J+1)]
        end
    end
    return newShare
end

## Draw from a Bayes Regression conditional on the joint distribution
## of the error process, and the other residual, either xi or upsilon.
## Takes a flattened versions of X in the input xHat.
## Nothing here is specific to β, runs for η as well.
function DrawBayesReg(data::Data, yVec::Matrix{Float64},
                      deltaVec::Vector{Float64}, util::Utility,
                      xHatMod::Vector{Float64},
                      exV::Vector{Float64}, sd::Vector{Float64},
                      cMap::Matrix{Int64}, xHat::Matrix{Float64},
                      priorMean, priorVarInv, K::Int64)

    counter = 1
    for t in 1:data.T
        for j in 1:data.J[t]
            ## This allows us to deconstruct the conditional distribution.
            xHatMod[counter] = 1.0 / (sd[cMap[t,j]])

            deltaVec[counter] = (yVec[t,j] -
                                 exV[counter])*xHatMod[counter]
            counter += 1
        end
    end

    xHat = xHat .* xHatMod;

    #Since we "observe" υ, that is information about ξ that needs to be used.
    betaTilde = (xHat'*xHat + priorVarInv) \
        (xHat'*deltaVec + priorVarInv*priorMean)

    varDraw = Hermitian(inv(xHat'*xHat + priorVarInv));
    F = cholesky(varDraw)

    betaDraw = betaTilde + F.L*rand( Normal(), K)

    return betaDraw
end

## Draw Γ conditional on shares, data and the distribution of errors.
## This relies on another inversion of the demand system, and begins
## to struggle in high dimension since the size of Γ is L(L+1).  If L
## is large, may need to condition on parts of Γ in order to explore
## the distribution effficiently.
function DrawGammaBLP(data::Data, util::Utility,
                      priors::PriorBLP, Γ::Matrix{Float64},
                      exV::Vector{Float64}, sd::Vector{Float64},
                      cMap::Matrix{Int64}, s::Matrix{Float64},
                      nSumers::Int64, pars::SearchParameters,
                      beta::Vector{Float64}, ζ)
    
    coinFlip = log(rand(Uniform()))
    ## The first L elements are non-zero
    oldGam = cholesky(Hermitian(Γ[1:data.L,1:data.L])).U

    ## We draw a candidate choleskly upper-triangular matrix to ensure
    ## positive-definite variance.
    candGamChol = zeros(data.L,data.L)
    for k in 1:data.L
        ## Diagonal elements of a cholesky U must be positive. Parameterize this.
        candGamChol[k,k] = exp(log(oldGam[k,k]) +
                               rand( Uniform(-pars.sigmaVarDiag,pars.sigmaVarDiag)))
        if pars.sigmaVarOffDiag > 0.0
            for j in (k+1):data.L
                candGamChol[k,j] = oldGam[k,j] + rand( Uniform(-pars.sigmaVarOffDiag,
                                                               pars.sigmaVarOffDiag))
            end
        end
        
    end

    ## This allows for arbitrary covariance between RC parameters,
    ## while also having terms remain in the B-H δ
    candGam = zeros(data.K,data.K)
    candGam[1:data.L,1:data.L] = candGamChol'*candGamChol

    ## Invert shares with new gamma. Continue as if α before. 
    for t in 1:data.T
        J = data.J[t]

        util.solutionFill[t,1:J] = FastInvertShares( data.X[t], s[t,1:J], nSumers,
                                                     candGam, ζ, J,
                                                     util.warmStart[t,1:J])
        
        util.Jac[t] = BuildXiJacBLP(data.X[t], util.solutionFill[t,1:J],
                                    nSumers, candGam, ζ, J)
        
        ξ = util.solutionFill[t,1:J] - data.X[t]*beta

        util.LMarketTemp[t] = MarketLikelihood( ξ, exV[cMap[t,1:J]], sd[cMap[t,1:J]],
                                                J, util.Jac[t])
    end
    ## Changing Γ conditional on shares only requires looking at the
    ## likelihood of xi.
    newLik = sum( util.LMarketTemp[t] for t in 1:data.T)
    oldLik = sum( util.LMarket[t] for t in 1:data.T)

    ## In addition to priors, our choice distribution q for the
    ## diagonals is not reversible. Use change of variable theorem here.
    for l in 1:data.L
        ## We don't need to worry about the support conditions for
        ## each, since the band never changes so both are within each
        ## other's support.
        newLik += -log( oldGam[l,l] )
        oldLik += -log( candGamChol[l,l])
    end
    
    
    ## Fill in the priors effect on the M-H computation.
    for l in 1:data.L
        newLik += logpdf( Gamma( priors.gammaDiagMu, priors.gammaDiagSigma),
                          candGamChol[l,l])
        oldLik += logpdf( Gamma( priors.gammaDiagMu, priors.gammaDiagSigma),
                          oldGam[l,l])
        for k in (l+1):data.L
            newLik += logpdf( Normal( priors.gammaOffDiagMu, priors.gammaOffDiagSigma),
                              candGamChol[l,k])
            oldLik += logpdf( Normal( priors.gammaOffDiagMu, priors.gammaOffDiagSigma),
                              oldGam[l,k])
        end
    end
    
    
    
    if coinFlip < newLik - oldLik
        newGam = candGam
        for t in 1:data.T
            util.LMarket[t] = util.LMarketTemp[t]
            for j in 1:data.J[t]
                util.warmStart[t,j] = util.solutionFill[t,j]
            end
        end
    else
        newGam = Γ
    end
    return newGam
end

## This replicates a Multivariate Regression from Rossi (2013) For a
## series of Covariates X and Vector outcomes Y. A, muBar, nu and V
## are priors.
function DoMultivariateReg(X::Matrix{Float64}, A::Matrix{Float64},
                           muBar::Vector{Float64}, Y::Matrix{Float64},
                           nu::Float64, V::Matrix{Float64})

    ## There are probably a lot of speed gains to be had here with
    ## some linear algebra. At the very least BTilde can be factored,
    ## and quadratic forms simplified.
    BTilde = (X'*X + A) \ (X'*Y + A*muBar')
    S = (Y - X'BTilde)'*(Y - X'*BTilde) + (BTilde - muBar')'*A*(BTilde - muBar')
            
    Σ = rand(InverseWishart(nu+1, Matrix(Hermitian(nu*V + S))))
    μ = rand(MvNormal(reshape(BTilde, (2)), Symmetric(Σ .* inv(X'*X + A))))

    return (Σ,μ)
end


## This function takes the series of residuals upsilon, xi and returns
## a mapping vMap which maps datapoints (xi, upsilon)_n to a
## distribution in the cluster, which is given by ( μ, Σ)_k. sigmaSize
## represents the size of the distribution, μ, Σ the vectors of
## distributions, V, nu, aMu, muBar are priors.
function DrawVMapDP(α::Float64, sigmaSize::Int64, N::Int64,
                    μ::Matrix{Float64}, Σ::Array{Float64,3},
                    upsilon::Vector{Float64}, xi::Vector{Float64},
                    vMap::Array{Int64}, V::Matrix{Float64}, nu::Float64,
                    aMu::Float64, muBar::Vector{Float64}, maxSigma::Int64)
    
    ## Get the counts for each nJ
    nJ = zeros(N);
    for n in 1:N
        nJ[vMap[n]] += 1
    end

    dat = hcat( upsilon, xi);

    R = cholesky(V).U
    logdetR = log( R[1,1] + R[2,2] )

    aMuFrac = aMu / (1.0 + aMu)

    ## R is already upper triangular cholesky so inv() isn't a bad call
    m = sqrt( aMuFrac )*inv(R)*( dat' .- muBar);

    ## The next 4 lines are code replicated from bayesm R package adapted
    ## from Rossi (2013) Code originally written by Wayne Taylor.
    vivi = [sum( m[:,k].^2 ) for k in 1:size(m,2)];

    base = log( (nu - 1) / 2.0) + log( aMuFrac ) - log( π ) - logdetR

    lnq0v = base .- ((nu-1) / 2.0) .* log.( 1.0 .+ vivi)
    q0 = exp.(lnq0v);

    ## Maximum possible size of sigmaSize is data.N
    probHolder = zeros(N+1);

    for n in 1:N
        probHolder .= 0.0
        datN = dat[n,:]

        ## We get the likelihood of the data coming from each cluster,
        ## as well as the probability it camefrom a new cluster (which
        ## is using the likelihood of a multivariate t-distribution)
        for i in 1:sigmaSize
            probHolder[i] = nJ[i]*pdf( MvNormal( μ[i,1:2], Σ[i,:,:]), datN)
        end
        probHolder[sigmaSize+1] = q0[n]*(α )

        ## Cap the number of possible distributions.
        if sigmaSize == maxSigma
            probHolder[sigmaSize+1] = 0.0
        end

        ## Floating point arithmatic can have problems here, if you
        ## see crashes after this line due to a lack of normalization,
        ## apply more robust normalizations using log properties.
        probHolder ./= sum(probHolder)

        ## These are necessary to manage the list
        oldVMap = vMap[n]
        fullCheck = nJ[oldVMap]

        ## Draw from a discrete distribution using a uniform against the inverse cdf.
        vMapUnif = rand(Uniform())
        cumSumPHolder = cumsum( probHolder)
        vMap[n] = 1
        for i in 1:length(probHolder)
            if vMapUnif < cumSumPHolder[i]
                vMap[n] = i
                break
            end
        end

        ## Now we have to do housekeeping with the list of distributions. We do this by cases
        oldSigmaSize = sigmaSize
        if vMap[n] == sigmaSize + 1
            ## Case 1: We add a new distribution

            ## Do a multivariate regression to get posterior means and
            ## variances conditional on this single data point, and
            ## our priors.
            X = ones(1,1)
            A = ones(1,1)*aMu

            
            ## If we were the only data point, then we replace that
            ## distribution, and don't change the total number
            if fullCheck == 1.0
                vMap[n] = oldVMap
                Σ[vMap[n],:,:],μ[vMap[n],:] = DoMultivariateReg(X, A,
                                                            muBar, dat[n:n,1:2],
                                                            nu, V)
            else
                Σ[vMap[n],:,:],μ[vMap[n],:] = DoMultivariateReg(X, A,
                                                                muBar, dat[n:n,1:2],
                                                                nu, V)
                ## Otherwise we've added a new distribution, draw its stuff.
                nJ[oldVMap] -= 1
                nJ[sigmaSize+1] = 1
                sigmaSize += 1
            end
        elseif fullCheck == 1.0 && oldVMap != vMap[n]
            ## We were the only data point in this distribution, and
            ## we're moving to an existing one.
            if oldVMap == sigmaSize
                nJ[oldVMap] = 0.0
                nJ[vMap[n]] += 1
                
                sigmaSize -= 1
            else## There is at least one occupied cluster with greater index
                ## If we are the only data point and switch, switch
                ## oldVMap with sigmaSize and decrement sigmaSize
                nJ[oldVMap] = nJ[sigmaSize]
                nJ[sigmaSize] = 0.0
                μ[oldVMap,:] = μ[sigmaSize,:]
                Σ[oldVMap,:,:] = Σ[sigmaSize,:,:]
                vMap[findall(x->x==sigmaSize,vMap)] .= oldVMap
                nJ[vMap[n]] += 1
                sigmaSize -= 1
            end
        else## Moving between clusters when no cluster must be deleted is trivial.
            nJ[oldVMap] -= 1
            nJ[vMap[n]] += 1
        end       
    end
    return vMap,μ,Σ,sigmaSize
end

## This function draws the joint distributions of (upsilon, xi) within
## cluster k, given vMap which places them in each cluster. Changing
## the distributions of the erros as well as the vMap in the previous
## step also requires that we recompute the likelihood of the shares
## of each market, so this function handles the housekeeping there.
function DrawSigma( upsilon, xi, sigmaSize, vMap,
                    muBar, aMu, sdXi, exVXi, sdUp, exVUp, data, util,
                    nSumers::Int64, Γ::Matrix{Float64},
                    β::Vector{Float64},
                    γ::Float64, α::Float64, BLP::Bool, ζ::Matrix{Float64})

    newSigma = zeros(sigmaSize,2,2)
    newMu = zeros(sigmaSize,2)

    ## Handle each cluster separately. 
    for i in 1:sigmaSize
        invVMap = findall( x->x==i, vMap)

        if length(invVMap ) == 0
            newSigma[i,:,:] = rand( InverseWishart( 5, [ 5.0  0.0; 0.0  5.0  ])  )
            continue
        end

        ## Notation follows Rossi (2013)
        sIV = hcat( upsilon[invVMap], xi[invVMap]  )
        nK = length(invVMap)

        ι = ones(nK);
        yBar = (1/nK)*sIV'*ι

        muTilde = (nK*yBar + aMu*muBar) / (nK + aMu)
        sIV -= ι*muTilde'
        v = 5 + nK
        ## Diffuse prior suggested in Rossi (2013)
        S = transpose(sIV)*sIV + aMu*(muTilde - muBar)*(muTilde - muBar)'
        V =  [ 5.0  0.0;
              0.0  5.0  ]

        ## Note that adding the prior can cause small floating point
        ## errors to fail symmetric or posdef checks required to draw
        ## from the InverseWishart Distribution, must cast to
        ## Symmetric and back to a matrix to prevent this.
        newSigma[i,:,:] = rand( InverseWishart( v, Matrix(Symmetric(V+S)))  )
        newMu[i,1:2] = rand( MvNormal(muTilde,(1.0 / (nK + aMu))*newSigma[i,:,:]))
        
        ## Conditional on upsilon, var and exV of ξ (Account for μ here as well)
        sdXi[invVMap] .= sqrt(newSigma[i,2,2] -
                              (newSigma[i,1,2]*newSigma[i,1,2] / newSigma[i,1,1]))
        exVXi[invVMap] = (newSigma[i,1,2] / newSigma[i,1,1])*(upsilon[invVMap] .- newMu[i,1]) .+
            newMu[i,2]
    end

    ## Now we need to update LMarket becuase Sigma and eta have
    ## changed.  Note that warmStart did not change, shares alpha beta
    ## and gamma are all that determine xi, so it remains constant
    ## through this step.
    for t in 1:data.T
        J = data.J[t]
        ξ = util.warmStart[t,1:J] - data.X[t]*β

        if J == 0
            util.LMarket[t] = 0.0
        else
            if BLP
                util.Jac[t] = BuildXiJacBLP(data.X[t], util.warmStart[t,1:J],
                                            nSumers, Γ, ζ, J)
            else
                util.Jac[t] = BuildXiJacBCS(data.X[t], util.solutionFill[t,1:J],
                                            γ, α, J)
            end
            
            util.shareJacLDet[t] = logabsdet(util.Jac[t])[1]
            util.LMarket[t] = MarketLikelihood( ξ, exVXi[data.cMap[t,1:J]],
                                                sdXi[data.cMap[t,1:J]],
                                                J, util.Jac[t])
        end
    end
    return newSigma, newMu, sdXi, exVXi
end

## This is the main function exported. Takes in data, priors/search,
## size of the chains (M, burnout), draws of the normal diswtribution
## (ζ), and initial values (S, δ, β, η, Γ, λ)
function DoMCMC( data::Data, searchParameters::SearchParameters,
                 M::Int64, ζ::Matrix{Float64}, util::Utility, priors::PriorBLP,
                 lamMap::Vector{Vector{Int64}}, maxMixtures::Int64, burnout::Int64,
                 S::Array{Float64,2}, δ::Array{Array{Float64,1},1}, β::Vector{Float64},
                 η::Vector{Float64}, Γ::Matrix{Float64}, λ::Vector{Float64})

    bigJ = maximum(data.J)
    α = priors.alpha

    ## Start with one mixture, let the DP handle it.
    sigmaSize = 1
    vMap = rand(Categorical(ones(1)), data.N);

    
    V = [1.0 0.0;
         0.0 1.0]
    nu = 5.0

    muBar = priors.muBar
    aMu = priors.aMu


    SkipNum = 2
    
    memAllocateSize = div( M - burnout, SkipNum )
    
    ## Allocate everything that the chains will fill in
    π = zeros(memAllocateSize,maxMixtures);
    Σ = zeros(memAllocateSize,maxMixtures,2,2);
    μ = zeros(memAllocateSize,maxMixtures,2);
    betaDraws = zeros(memAllocateSize,data.K);
    etaDraws = zeros(memAllocateSize,data.numInst);
    xi = zeros(memAllocateSize,data.N);
    upsilon = zeros(memAllocateSize,data.N);
    lamT = zeros(memAllocateSize,data.T);
    shareDraws = zeros(memAllocateSize,data.T,bigJ+1);
    gammaDraws = zeros(memAllocateSize,data.K,data.K);

    ## These are flattened versions of the prices, X, and Z used in
    ## the regressions for computational ease.
    pMat = zeros(data.T,bigJ);
    xHat = zeros(data.N,data.K);
    zHat = zeros(data.N,data.numInst);

    flatDelta = zeros(data.N);
    flatPrice = zeros(data.N);
    nSumers = 25

    curSigma = zeros(maxMixtures,2,2);
    newSigma = zeros(maxMixtures,2,2);
    curMu = zeros(maxMixtures,2);
    newMu = zeros(maxMixtures,2);
    curBeta = zeros(data.K);
    newBeta = zeros(data.K);
    curEta = zeros(data.numInst);
    newEta = zeros(data.numInst);
    curXi = zeros(data.N);
    newXi = zeros(data.N);
    curUpsilon = zeros(data.N);
    newUpsilon = zeros(data.N);
    curlamT = zeros(data.T);
    newLamT = zeros(data.T);
    curShare = zeros(data.T,bigJ+1);
    newShare = zeros(data.T,bigJ+1);
    curGam = zeros(data.K,data.K);
    newGam = zeros(data.K,data.K);


    
    
    ## Find Starting points
    for i in 1:maxMixtures
        curSigma[i,:,:] = [1.0 0.0; 0.0 1.0]
    end

    ## Without any panel structure to exploit as in the airline
    ## example, our starting values will not be so great, so we pass
    ## them in as arguements.
    curGam= Γ
    curBeta = β
    curETa = η
    curShare = S
    curlamT = λ
        
    saveCount = 1

    ## We now fill in the flattened matrices, the holder for delta
    ## (util.warmStart), previous Jacs and Likelihoods as well.
    for t in 1:data.T

        xHat[data.cMap[t,1:data.J[t]],:] = data.X[t]
        zHat[data.cMap[t,1:data.J[t]],:] = data.Z[t]
        pMat[t,1:data.J[t]] = data.X[t][:,1]
        flatPrice[data.cMap[t,1:data.J[t]]] = pMat[t,1:data.J[t]]
        
        util.warmStart[t,1:data.J[t]] = δ[t][1:data.J[t]]

        if PredictSharesBLP( δ[t][1:data.J[t]], data.X[t], 25, Γ, ζ ) != S[t,1:data.J[t]]
            println("t bad: $t")
        end
        
        

        flatDelta[data.cMap[t,1:data.J[t]]] = util.warmStart[t,1:data.J[t]]

        util.Jac[t] = BuildXiJacBLP( data.X[t], util.warmStart[t,1:data.J[t]], nSumers,
                                     curGam, ζ, data.J[t] )

        util.LMarket[t] = MarketLikelihood( util.warmStart[t,1:data.J[t]] -
                                            data.X[t]*curBeta,
                                       zeros(data.J[t]), ones(data.J[t]),
                                       data.J[t], util.Jac[t])
    end

    ## Fill in the residuals using the initialized values of xi, upsilon
    for t in 1:data.T
        curXi[data.cMap[t,1:data.J[t]]] = util.warmStart[t,1:data.J[t]] -
            data.X[t]*newBeta
        curUpsilon[data.cMap[t,1:data.J[t]]] = pMat[t,1:data.J[t]] -
            data.Z[t]*newEta
    end

    ## These are storages used for expressing the conditional
    ## distribution of xi, upsilon quickly.
    deltaVecHolder = zeros(data.N);
    xHatMod = zeros(data.N);
    sdXi = ones(data.N);
    exVXi = zeros(data.N);
    sdUp = ones(data.N);
    exVUp = zeros(data.N);


    ## For whatever initialzation of vMap, fill it with reduced form
    ## estimates of the distributions.
    for i in 1:sigmaSize
        invVMap = findall( x->x==i, vMap);

        curSigma[i,:,:] = [var(curUpsilon[invVMap]) cov( curUpsilon[invVMap], curXi[invVMap] );
                           cov( curUpsilon[invVMap], curXi[invVMap] ) var(curXi[invVMap])]

        curMu[i,1] = mean( curUpsilon[invVMap])
        curMu[i,2] = mean( curXi[invVMap])

        sdXi[invVMap] .= sqrt(curSigma[i,2,2] -
                              (curSigma[i,1,2]*curSigma[i,1,2] / curSigma[i,1,1]))
        exVXi[invVMap] = (curSigma[i,1,2] / curSigma[i,1,1])*(curUpsilon[invVMap] .-
                                                              curMu[i,1]) .+
                                                              curMu[i,2]

        ## We need the conditional expectations and variances of upsilon too
        sdUp[invVMap] .= sqrt(curSigma[i,1,1] -
                              (curSigma[i,1,2]*curSigma[i,1,2] / curSigma[i,2,2]))
        exVUp[invVMap] = (curSigma[i,1,2] / curSigma[i,2,2])*(curXi[invVMap] .- curMu[i,2]) .+
            curMu[i,1]
    end


    println("Begin Chain")
    for m in 1:(M-1)
        if m % 100 == 0
            println("m = $m, sigmaSize = $sigmaSize")
        end

        ## THis is the main estimation loop. Drawing each parameter in
        ## a Hybrid-Gibbs fashion.

        newLamT = DrawLambda( data, curlamT, newLamT, curShare,
                                  searchParameters, lamMap, priors)

        newShare =  DrawShares( data, curShare, newLamT,
                                curBeta, util, newShare,
                                searchParameters, exVXi, sdXi,
                                data.cMap, nSumers,
                                curGam, ζ);



        newGam = DrawGammaBLP( data, util, priors, curGam,
                               exVXi, sdXi, data.cMap,
                               newShare, nSumers,
                               searchParameters, curBeta,
                               ζ)

        
        
        newBeta = DrawBayesReg( data, util.warmStart, deltaVecHolder,
                                util, xHatMod, exVXi, sdXi, data.cMap,
                                xHat, priors.betaMean, priors.betaVarInv,
                                data.K )

        ## After drawing beta, we have changed the values of xi, and
        ## need to update that through so we can then draw eta.
        for t in 1:data.T
            newXi[data.cMap[t,1:data.J[t]]] = util.warmStart[t,1:data.J[t]] -
                data.X[t]*newBeta
        end
        
        

        for i in 1:sigmaSize
            invVMap = findall( x->x==i, vMap);
            ## We need the conditional expectations and variances of upsilon too
            sdUp[invVMap] .= sqrt(curSigma[i,1,1] -
                                  (curSigma[i,1,2]*curSigma[i,1,2] / curSigma[i,2,2]))
            exVUp[invVMap] = (curSigma[i,1,2] / curSigma[i,2,2])*(newXi[invVMap] .- curMu[i,2]) .+
                curMu[i,1]
        end
        

        newEta = DrawBayesReg( data, pMat, deltaVecHolder, util, xHatMod, exVUp, sdUp, data.cMap, zHat, priors.etaMean, priors.etaVarInv, data.numInst )

        ## Note that LMarket is no longer valid until DrawSigma changes it.
        for t in 1:data.T
            newUpsilon[data.cMap[t,1:data.J[t]]] = pMat[t,1:data.J[t]] -
                data.Z[t]*newEta
        end
                    
        vMap,curMu,curSigma,sigmaSize = DrawVMapDP( α, sigmaSize, data.N,
                                                    curMu, curSigma,
                                                    newUpsilon, newXi,
                                                    vMap, V, nu,
                                                    aMu, muBar,
                                                    maxMixtures);

       newSigma[1:sigmaSize,:,:], newMu[1:sigmaSize,:], sdXi, exVXi =
            DrawSigma( newUpsilon, newXi, sigmaSize, vMap, muBar, aMu,
                       sdXi, exVXi, sdUp, exVUp, data, util, nSumers,
                       newGam,
                       newBeta, 0.0, 0.0, true, ζ);

        ## After completing a pass of the loop, copy the new
        ## parameters into the current parameters, and save if needed.
        curSigma = copy(newSigma)
        curMu = copy(newMu)
        curBeta = copy(newBeta)
        curEta = copy(newEta)
        curXi = copy(newXi)
        curUpsilon = copy(newUpsilon)
        curlamT = copy(newLamT)
        curShare = copy(newShare)
        curGam = copy(newGam)

        if m > burnout && m % SkipNum == 0
            for n in 1:data.N
                π[saveCount,vMap[n]] += 1.0 / data.N
            end
            Σ[saveCount,1:maxMixtures,1:2,1:2] = newSigma
            μ[saveCount,1:maxMixtures,1:2] = newMu
            betaDraws[saveCount,1:data.K] = newBeta
            etaDraws[saveCount,1:data.numInst] = newEta
            xi[saveCount,1:data.N] = newXi
            upsilon[saveCount,1:data.N] = newUpsilon
            lamT[saveCount,1:data.T] = newLamT
            shareDraws[saveCount,1:data.T,1:(bigJ+1)] = newShare
            gammaDraws[saveCount,1:data.K,1:data.K] = newGam
            saveCount += 1
            
        end

    end
    return betaDraws, etaDraws, μ, Σ, π, xi, upsilon, lamT, shareDraws, gammaDraws
end



