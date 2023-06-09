using JLD2
using DataFrames
using CSV
using Plots
pyplot()

include("bayes_BLP_DP.jl")


## Loads searchParameters stored at ../SimDir/Args[1]Args[3].jld2 and
## clones it over to ../SimDir/Args[1](1:Args[2]).jld2

simName = ARGS[1]
nSims = parse(Int, ARGS[2])
Clone = parse(Int, ARGS[3])

fileName = "../SimDir/$(simName)$(Clone).jld2"

@load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

cloneSeachParams = SearchParameters( searchParameters.sigmaVarDiag,
                                     searchParameters.sigmaVarOffDiag,
                                     searchParameters.dScale,
                                     searchParameters.lambdaTVar,
                                     searchParameters.lambdaDVar,
                                     searchParameters.gammaVar,
                                     searchParameters.alphaVar)

for i in 1:nSims
    fileName = "../SimDir/$(simName)$(i).jld2"

    @load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

    searchParameters = SearchParameters( cloneSeachParams.sigmaVarDiag,
                                         cloneSeachParams.sigmaVarOffDiag,
                                         cloneSeachParams.dScale,
                                         cloneSeachParams.lambdaTVar,
                                         cloneSeachParams.lambdaDVar,
                                         cloneSeachParams.gammaVar,
                                         cloneSeachParams.alphaVar)

    @save "../SimDir/$(simName)$(i).jld2" data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ
    println("Adjusted $fileName")
end
