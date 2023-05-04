using Distributions
using LinearAlgebra
using JLD2

include("bayes_BLP_DP.jl")

function RunEstim( fileName, nMixtures)

    @load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

    ## Allocate the memory that will be used to fill Utility()
    denom = zeros(2)
    bigJ = maximum(data.J)
    shareHolder = zeros(bigJ)
    solutionFill = zeros(data.T,bigJ);

    LMarket = zeros(data.T);
    LMarketTemp = zeros(data.T);

    shareJacLDet = zeros(data.T);
    Jac = Vector{Matrix{Float64}}(undef,data.T);
    sJac = Vector{Matrix{Float64}}(undef,data.T);
    warmStart = zeros(data.T,bigJ);
    for t in 1:data.T
        Jac[t] = zeros(data.J[t],data.J[t])
    end
    delta = zeros(bigJ)

    util = Utility( denom, shareHolder, solutionFill, LMarket, LMarketTemp, Jac, sJac, delta,
                    warmStart,  shareJacLDet);

    ## Set M and the burnout 
    M = 6000

    burnout = 4000

    ## Run the MCMC Estimation Routine using the stored data.
    betaDraws, etaDraws, μ, Σ, π, xi, upsilon,lamT, shareDraws, gammaDraws =
        DoMCMC(data, searchParameters, M, ζ, util, priors, lamMap, nMixtures, burnout,
               S, δ, βSim, ηSim, Γ, λ);

    ## Output to stdout some statistics about the parameters.
    betaMean = [mean( betaDraws[:,k]) for k in 1:data.K]
    println( "betaMean: $betaMean")
    etaMean = [mean( etaDraws[:,k]) for k in 1:data.numInst]
    println( "etaMean: $etaMean")
    μMean = [mean( μ[:,k,i]) for k in 1:nMixtures, i in 1:2]
    println( "μMean: $μMean")
    ΣMean = [mean(Σ[:,k,i,j]) for i in 1:2, j in 1:2, k in 1:nMixtures]
    println( "ΣMean: $ΣMean")
    πMean = [mean( π[:,k]) for k in 1:nMixtures]
    println( "πMean: $πMean")
    lamTMean = [mean( exp.(lamT[:,t])) for t in 1:data.T]
    println( "lamTMean: $lamTMean")
    shareTMean = [sum( mean( shareDraws[m,t,j] for m in 1:(size(shareDraws,1)-1))
                       for j in 1:data.J[t]) for t in 1:data.T]
    println( "shareTMean: $shareTMean")

    gamMean = [mean( gammaDraws[:,i,j]) for i in 1:data.K, j in 1:data.K]
    println( "gamMean: $gamMean")

    return betaMean, etaMean, μMean, ΣMean, πMean, lamTMean, shareTMean, gamMean, betaDraws, etaDraws, μ, Σ, π, xi, upsilon,lamT, shareDraws, gammaDraws
end


tempArgs = copy(ARGS)

display( tempArgs )

## Load commandline arguments and run estimation given the inputs.
fileName = tempArgs[1]
maxMixtures = parse(Int, tempArgs[2])
outputName = tempArgs[3]

tmp = RunEstim( fileName, maxMixtures);

## Save output to the third command line arg.
@save outputName tmp
