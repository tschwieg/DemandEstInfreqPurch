using JLD2
using Printf
using CSV
using DataFrames
include("bayes_BLP_DP.jl")


## Helper functions for displaying strings
function cln(x)
    return @sprintf "%.3f" x
end

function fixPar(x)
    return x[1:2] * "\\%"
end
## Specs are hardcoded as well as their data locations.
specs = ["J25/FONCPrice_big25J", "J45/FONCPrice_big45J", "J25/FONCPrice_sml25J", "J25/FONCPrice_over25J", "J25/FONCPrice_sin25J", "J3/FONCPrice_high3J"]
specData = ["J25/FONCPrice_big25J","J45/FONCPrice_big45J", "J25/FONCPrice_sml25J", "J25/FONCPrice_over25J", "J25/FONCPrice_sin25J", "J3/FONCPrice_high3J"]


##Sizes for TableOutput
nRows = 7
nCols = length(specs)
colIndex = 0

TableOutput = Matrix{String}(undef, nRows, nCols*7)#zeros(nRows, nCols*4  )
TableOutput .= ""


M = 999

nSims = 100

L = 1
K = 2
numInst = 2

bigJ = 45


for (q,spec) in enumerate(specs)

    println(spec)

    trueBetaSim = zeros(nSims,K)
    trueEtaSim = zeros(nSims,K)
    trueGamSim = zeros(nSims,K)
    
    ## Load one sepc of data so we have the sizes correct.
    @load "../SimDir/"*specData[q]*"1.jld2" data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ
    
    global colIndex
    betaDraws = zeros(nSims,M,K);
    etaDraws = zeros(nSims,M,numInst);
    gammaDraws = zeros(nSims,M,K,K);
    lamDraws = zeros(nSims,M,data.T);

    ## Load the Data and store it in the required tables
    for i in 1:nSims
        @load "../OutputDir/FONC/$(spec)$(i).jld2" tmp
        betaDraws[i,:,:] = copy(tmp[9][1:M,1:K]);
        etaDraws[i,:,:] = copy(tmp[10][1:M,1:numInst]);
        gammaDraws[i,:,:,:] = copy(tmp[end][1:M,1:K,1:K]);
        lamDraws[i,:,:] = copy(tmp[end-2][1:M,1:data.T]);
    end

    betaSamples = zeros(nSims,K,3);
    etaSamples = zeros(nSims,numInst,3);
    gamSamples = zeros(nSims,K,K,3);
    lamSamples = zeros(nSims,3);
    betaContained = zeros(nSims,K);
    etaContained = zeros(nSims,numInst);
    gammaContained = zeros(nSims,K);
    lamContained = zeros(nSims);
    for i in 1:nSims
        betaSamples[i,:,1] = [quantile(betaDraws[i,1:M,k], .025 ) for k in 1:K]
        betaSamples[i,:,2] = [mean(betaDraws[i,m,k] for m in 1:M) for k in 1:K]
        betaSamples[i,:,3] = [quantile(betaDraws[i,1:M,k], .975 ) for k in 1:K]

        etaSamples[i,:,1] = [quantile(etaDraws[i,1:M,k], .025 ) for k in 1:numInst]
        etaSamples[i,:,2] = [mean(etaDraws[i,m,k] for m in 1:M) for k in 1:numInst]
        etaSamples[i,:,3] = [quantile(etaDraws[i,1:M,k], .975 ) for k in 1:numInst]

        gamSamples[i,:,:,1] = [quantile(gammaDraws[i,1:M,k,l], .025 ) for k in 1:K, l in 1:K]
        gamSamples[i,:,:,2] = [mean(gammaDraws[i,m,k,l] for m in 1:M) for k in 1:K, l in 1:K]
        gamSamples[i,:,:,3] = [quantile(gammaDraws[i,1:M,k,l], .975 ) for k in 1:K, l in 1:K]

        b = reshape( lamDraws[i,1:M,1:data.T], M*data.T)
        lamSamples[i,1] = quantile( b, .025 )
        lamSamples[i,2] = mean(b)
        lamSamples[i,3] = quantile(b, .975 )
    end

    
    

    for k in 1:K
        b = zeros(7)
        b[1] = (mean(betaSamples[:,k,2])-βSim[k])

        b[2] = quantile( betaSamples[:,k,2], .025 ) - βSim[k]
        b[3] = quantile( betaSamples[:,k,2], .975 ) - βSim[k]
        b[4] = (quantile(betaSamples[:,k,2], .5) - βSim[k])
        b[5] = median( abs.(betaSamples[:,k,2] .- βSim[k] ) )
        b[6] = mean( abs.(betaSamples[:,k,2] .- βSim[k] ))
        b[7] = mean((betaSamples[:,k,2] .- βSim[k] ).^2)#var( betaSamples[:,k,2] )
        for i in 1:7
            
            if b[i] <= 0.0 
                TableOutput[k,colIndex+i] = @sprintf "\$%.3f\$" b[i]
            else
                TableOutput[k,colIndex+i] = @sprintf "\$\\ %.3f\$" b[i]
            end
        end

    end

    for k in 1:numInst
        b = zeros(7)

        b[1] = (mean(etaSamples[:,k,2])-ηSim[k])

        b[2] = quantile( etaSamples[:,k,2], .025 ) - ηSim[k]
        b[3] = quantile( etaSamples[:,k,2], .975 ) - ηSim[k]
        
        b[4] = (quantile(etaSamples[:,k,2], .5) - ηSim[k])
        b[5] = median( abs.(etaSamples[:,k,2] .- ηSim[k] ) )
        b[6] = mean( abs.(etaSamples[:,k,2] .- ηSim[k] ) )
        b[7] = mean((etaSamples[:,k,2] .- ηSim[k] ).^2)
        for i in 1:7
            if b[i] <= 0.0 
                TableOutput[K+k,colIndex+i] = @sprintf "\$%.3f\$" b[i]
            else
                TableOutput[K+k,colIndex+i] = @sprintf "\$\\ %.3f\$" b[i]
            end
        end

    end

    kCounter = K+numInst

    for k in 1:L
        b = zeros(7)

        b[1] = (mean(gamSamples[:,k,k,2])-Γ[k,k])

        b[2] = quantile( gamSamples[:,k,k,2], .025 ) - Γ[k,k]
        b[3] = quantile( gamSamples[:,k,k,2], .975 ) - Γ[k,k]
        

        b[4] = (quantile(gamSamples[:,k,k,2], .5) - Γ[k,k])
        b[5] = median( abs.(gamSamples[:,k,k,2] .- Γ[k,k] ))
        b[6] = mean( abs.(gamSamples[:,k,k,2] .- Γ[k,k] ))
        b[7] = mean((gamSamples[:,k,k,2] .- Γ[k,k] ).^2)
        for i in 1:7
            if b[i] <= 0.0 
                TableOutput[kCounter+k,colIndex+i] = @sprintf "\$%.3f\$" b[i]
            else
                TableOutput[kCounter+k,colIndex+i] = @sprintf "\$\\ %.3f\$" b[i]
            end
        end

    end
    
    kCounter += L

    b = zeros(7)

    b[1] = (mean(lamSamples[:,2]).-λ[1])
    b[2] = quantile( lamSamples[:,2], .025 ) -λ[1]
    b[3] = quantile( lamSamples[:,2], .975 ) -λ[1]
    b[4] = (quantile(lamSamples[:,2], .5) - λ[1])
    b[5] = median( abs.(lamSamples[:,2] .- λ[1]) )
    b[6] = mean( abs.(lamSamples[:,2] .- λ[1]) )
    b[7] = mean((lamSamples[:,2] .- λ[1] ).^2)
    for i in 1:7
        if b[i] <= 0.0 
            TableOutput[kCounter+1,colIndex+i] = @sprintf "\$%.3f\$" b[i]
        else
            TableOutput[kCounter+1,colIndex+i] = @sprintf "\$\\ %.3f\$" b[i]
        end
    end

    kCounter += 1

    percentZeros = zeros(nSims);
    for i in 1:nSims
        @load "../SimDir/"*specData[q]*"$i.jld2" data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ

        percentZeros[i] = 1.0 - sum(data.q .!= 0) / sum(data.J)
    end
    b = zeros(4)
    b[1] = mean(percentZeros)

    b[2] = quantile(percentZeros, .025)
    b[3] = quantile(percentZeros, .975)
    b[4] = quantile(percentZeros, .5)
    for i in 1:4
        
        if b[i] <= 0.0 
            TableOutput[kCounter+1,colIndex+i] = @sprintf "\$%.3f\$" b[i]
        else
            TableOutput[kCounter+1,colIndex+i] = @sprintf "\$\\ %.3f\$" b[i]
        end
    end
    colIndex += 7

end


rowNames = ["\$\\alpha\$", "\$\\beta_1\$", "\$\\eta_1\$","\$\\eta_2\$",
            "\$\\Gamma_{11}\$","\$\\lambda\$", "0 Distribution" ]


colHeaders = vcat(repeat(["25J"], 7),
                  repeat(["45J"], 7),
                  repeat(["Small_Lam_25J"], 7),
                  repeat(["Over_25J"], 7),
                  repeat(["Sin25J"], 7),
                  repeat(["J3"], 7))


colNames = vcat( repeat( ["Mean", "2.5Pct.", "97.5Pct.", "Median", "Median Abs Bias", "Mean Absolute Deviation", "MSE"], nCols ) )

a = convert( Vector{String}, ( vcat( ["Row Label"], colNames .* " - " .* colHeaders )) )

vcat( permutedims( vcat( [""], colNames .* " - " .* colHeaders )), hcat( rowNames, TableOutput) )

newdf = convert(DataFrame, hcat( rowNames, TableOutput))
rename!(newdf, names(newdf) .=> Symbol.(a))

CSV.write( "../OutputDir/simResults.csv", newdf)
