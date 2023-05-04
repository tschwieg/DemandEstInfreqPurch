using JLD2
using DataFrames
using CSV
using Plots
pyplot()

include("bayes_BLP_DP.jl")

simName = ARGS[1]
nSims = parse(Int, ARGS[2])
groupBy = parse(Int64, ARGS[3])
simDirName = ARGS[4]

@load "../OutputDir/$(simName)1.jld2" tmp

M = size( tmp[9], 1)
K = size( tmp[9], 2)
numInst = size( tmp[10],2)
T = size( tmp[end-2], 2)
bigJ = size(tmp[end-1],3)

#shareDraws = zeros(nSims,M,T,bigJ);
#gammaDraws = zeros(nSims,M,K,K);

shareAcceptance = zeros(nSims)
gamAcceptance = zeros(nSims)

validHists = []

for i in 1:nSims

    fName = "../OutputDir/$(simName)$(i).jld2"
    if !isfile( fName )
        continue
    end
    
    @load fName tmp

    gammaDraws = copy(tmp[end][1:M,1:K,1:K]);
    shareDraws = copy(tmp[end-1][1:M,1:T,1:bigJ]);

    gamAcceptance[i] = length( unique( gammaDraws[:,1,1])) / length( gammaDraws[:,1,1])
    shareAcceptance[i] = length( unique( shareDraws[:,:,1])) / length( shareDraws[:,:,1])

    push!( validHists, i)
end

nGroups = div( nSims, groupBy)
for i in 1:nGroups

    iter = Iterators.collect( filter( x->isfile( "../OutputDir/$(simName)$(x).jld2"),
                                      ((i-1)*groupBy+1):(i*groupBy)))

    if length(iter) == 0
        continue
    end
    

    fileName = "../SimDir/$(simDirName)$(iter[1]).jld2"
    @load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ
    
    println( "Group $i Gam Acceptance: $(mean(gamAcceptance[j] for j in iter))")
    println( "         Gam Search Param: $(searchParameters.sigmaVarDiag)")
    println( "         Share Acceptance: $(mean(shareAcceptance[j] for j in iter))")
    println( "         Share Search Param: $(searchParameters.dScale)")
    println( "--------------------------------------------------------------------")
    println("")
end


histogram( gamAcceptance[validHists], title="Gamma Acceptance Rate across Sims",
           label="")
savefig("DiagnosticPlots/$(simName)_Gam.pdf")

histogram( shareAcceptance[validHists], title="Share Acceptance Rate across Sims",
           label="")
savefig("DiagnosticPlots/$(simName)_Share.pdf")
