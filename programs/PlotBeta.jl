using JLD2
using DataFrames
using CSV
using Plots
pyplot()

include("bayes_BLP_DP.jl")

simName = ARGS[1]
nSims = parse(Int, ARGS[2])
estName = ARGS[3]
outputLoc = ARGS[4]


betaMeans = zeros(nSims,1)

for i in 1:nSims
    fileNameSim = "../SimDir/$(simName)$(i).jld2"
    fileNameEst = "../OutputDir/FONC/$(estName)$(i).jld2"


    @load fileNameSim data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

    @load fileNameEst tmp
    betaDraws = copy(tmp[9]);

    betaMeans[i,1] = mean(betaDraws[2,:]) - βSim[2]
end

name = [Symbol(simName*" Beta1")]
newdf = convert(DataFrame, betaMeans)
rename!(newdf, names(newdf) .=> name)

CSV.write( "../OutputDir/$(outputLoc)/$(simName)_Beta1_Hist.csv", newdf)
