using JLD2
using DataFrames
using CSV
using Plots
pyplot()

include("bayes_BLP_DP.jl")

simName = ARGS[1]
nSims = parse(Int, ARGS[2])
gamAdjust = parse(Float64, ARGS[3])
dScaleAdjust = parse(Float64, ARGS[4])
minIndex = parse(Int64, ARGS[5])
maxIndex = parse(Int64, ARGS[6])

for i in minIndex:maxIndex
    fileName = "../SimDir/$(simName)$(i).jld2"

    @load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

    searchParameters = SearchParameters( searchParameters.sigmaVarDiag*gamAdjust,
                                         searchParameters.sigmaVarOffDiag,
                                         searchParameters.dScale*dScaleAdjust,
                                         searchParameters.lambdaTVar,
                                         searchParameters.lambdaDVar,
                                         searchParameters.gammaVar,
                                         searchParameters.alphaVar)

    @save "../SimDir/$(simName)$(i).jld2" data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ
    println("Adjusted $fileName")
end


    
