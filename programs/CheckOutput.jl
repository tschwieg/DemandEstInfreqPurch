using JLD2
using DataFrames
using CSV
using Plots
pyplot()

include("bayes_BLP_DP.jl")

simName = ARGS[1]
nSims = parse(Int, ARGS[2])
simDirName = ARGS[3]

@load "../OutputDir/$(simName)1.jld2" tmp

M = size( tmp[9], 1)
K = size( tmp[9], 2)
numInst = size( tmp[10],2)
T = size( tmp[end-2], 2)
bigJ = size(tmp[end-1],3)

#shareDraws = zeros(nSims,M,T,bigJ);
#gammaDraws = zeros(nSims,M,K,K);

betaMeans = zeros(nSims,K)
gammaMeans = zeros(nSims,1)
etaMeans = zeros(nSims,numInst)
betaCoverage = zeros(K)
gammaCoverage = zeros(1)
etaCoverage = zeros(numInst)


for i in 1:nSims

    @load "../OutputDir/$(simName)$(i).jld2" tmp

    gammaDraws = copy(tmp[end][1:M,1,1]);
    betaDraws = copy(tmp[9][1:M,1:K]);
    etaDraws = copy(tmp[10][1:M,1:numInst]);

    fileName = "../SimDir/$(simDirName)$(i).jld2"
    @load fileName data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

    for k in 1:K
        betaCoverage[k] += (1.0 / nSims)*( βSim[k] >= quantile( betaDraws[:,k], .025) &&
                                           βSim[k] <= quantile( betaDraws[:,k], .975) )
        betaMeans[i,k] = mean( betaDraws[:,k] )
    end

    gammaCoverage[1] += (1.0 / nSims)*( Γ[1,1] >= quantile( gammaDraws[:,1,1], .025) &&
                                        Γ[1,1] <= quantile( gammaDraws[:,1,1], .975) )
    gammaMeans[i,1] = mean( gammaDraws[:,1,1])

    for k in 1:numInst
        etaCoverage[k] += (1.0 / nSims)*( ηSim[k] >= quantile( etaDraws[:,k], .025) &&
                                           ηSim[k] <= quantile( etaDraws[:,k], .975) )
        etaMeans[i,k] = mean( etaDraws[:,k] )
    end    
end

for k in 1:1
    histogram( betaMeans[:,k], title="\$\\beta[$(k)]\$ Posterior Mean Across Simulations",
               label="")
    savefig("DiagnosticPlots/$(simName)_Mean_Beta$(k).pdf")
end

histogram( gammaMeans[:,1], title="\$\\Gamma[1]\$ Posterior Mean Across Simulations")
savefig("DiagnosticPlots/$(simName)_Mean_Gamma1.pdf")


for k in 1:numInst
   histogram( etaMeans[:,k], title="\$\\eta[$(k)]\$ Posterior Mean Across Simulations",
               label="")
    savefig("DiagnosticPlots/$(simName)_Mean_Eta$(k).pdf")
end

for k in 1:K
    println("Beta[$(k)] Coverage: $(betaCoverage[k])")
end
for k in 1:numInst
    println("Eta[$(k)] Coverage: $(etaCoverage[k])")
end
println("Gamma[1] Coverage: $(gammaCoverage[1])")
