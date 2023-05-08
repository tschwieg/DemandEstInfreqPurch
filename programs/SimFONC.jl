using Distributions
using LinearAlgebra
using Plots
using Optim
using JLD2

include("bayes_BLP_DP.jl")

## This function simulates T markets with characteristics drawn from
## XDist, priced according to a simultaneous Pricing Nash-Betrand
## Equilibrium, where there is no shared ownership.
## J Dist: Discrete Distribution over the number of players in the market
## zDists: numInst continuous distributions giving the cost shifters for each firm
## XDist: 1 continuous distributions giving the covirates for each firm.
## arrivalsVec: T discrete distributions of arrivals for each market.
## Γ, βSim, ηSim, ζ are preferences, errors are draws of xi,
## deltaAdjust is deprecated, unused in FONC spec.
function BuildDataFONC( T::Int64, JDist, K::Int64, L::Int64, Γ::Matrix{Float64},
                    βSim::Vector{Float64}, ηSim::Vector{Float64},
                    errors::Vector{Float64}, numInst::Int64,
                    zDists, XDist, arrivalsVec,
                    ζ::Matrix{Float64}, deltaAdjust::Float64)

    J = rand( JDist, T)
    bigJ = maximum(J)
    nSumers = 25

    ## Allocate the various objects that will go into the Data struct.
    cMap = Matrix{Int64}(undef,T,bigJ);
    X = Vector{Matrix{Float64}}(undef,T);
    Z = Vector{Matrix{Float64}}(undef,T);
    A = Vector{Int64}(undef,T);
    q = Matrix{Float64}(undef,T,bigJ);
    δ = Vector{Vector{Float64}}(undef,T);

    ## Size of all data points is the number of products in total.
    N = sum(J)
    S = zeros(T,bigJ+1);

    q .= 0.0;
    counter = 1

    for t in 1:T
        ## For each market, keep a reference to how [1:T,1:J[t]] maps to 1:N
        cMap[t,1:J[t]] = counter:(counter+J[t]-1)
        X[t] = zeros(J[t],K)

        ## Fill in cost-shifters
        Z[t] = zeros(J[t],numInst)
        for z in 1:numInst
            Z[t][:,z] = rand( zDists[z], J[t])
        end

        ## X is filled from a single distribution, we only use
        ## Multinomial() in our sims.
        X[t][:,2:end] = rand( XDist, J[t])[2:end,:]'



        MC = Z[t]*ηSim
        ## Need to solve the equation:
        ## D + J_p (p - mc) = 0
        del(p) = X[t][:,2:end]*βSim[2:end] + errors[cMap[t,1:J[t]]] + p*βSim[1]
        xFun(p) = hcat( p, X[t][:,2:end] )
        Dfun(p) = PredictSharesBLP( del(p), xFun(p), nSumers, Γ, ζ)


        Jp(p) = BuildPriceJacBLP( xFun(p), del(p), nSumers, Γ, ζ, J[t], βSim )
        Imat = diagm(ones(J[t]))
                
        loss(p) = sum((Dfun(p) + (Imat .* Jp(p))*(p - MC)).^2)
        oLoss(p) = -sum( ((p - MC) .* Dfun(p)).^2)

        ## Rather than solving the system, we minimize the SSR of the system.
        a = optimize(loss, ones(J[t]), BFGS(), autodiff=:forward)

        ## a.minimizer are optimal prices.
        if any( a.minimizer .>= 10.0 )
            println("Market $t has prices above 10.0")
            println("X: ", X[t])
            println("ξ: ", errors[cMap[t,1:J[t]]])
            println("MC:  ", MC)
            println("P:  ", a.minimizer)
        end
        
        ## Set prices by the FONCs
        X[t][:,1] = a.minimizer
       
        ## Once prices are known, set δ, then predict shares and simulate arrivals.
        δ[t] = X[t]*βSim + errors[cMap[t,1:J[t]]]
        S[t,1:J[t]] = PredictSharesBLP( δ[t], X[t], nSumers, Γ, ζ)
        S[t,J[t]+1] = 1.0 - sum( S[t,1:J[t]] )
        A[t] = rand( arrivalsVec[t])

        ## A is only a signal of the arrivals process, not the actual
        ## observed arrivals (since otherwise we would use multinomial
        ## demand rather than poisson demand)
        ActualArrivals = rand( arrivalsVec[t])
        q[t,1:J[t]] = rand( Multinomial(ActualArrivals, S[t,1:(J[t]+1)] ))[1:J[t]]
        counter += J[t]
    end


    return Data( X, q, Z, J, T, A, N, K, L, numInst, cMap), S, δ;
end


## Set the size of the market, as well as the distributional
## constraints we wil use to pass to data creation.
T = parse( Int64, ARGS[3], base = 10)

minJ = parse( Int64, ARGS[4], base = 10)
maxJ = parse( Int64, ARGS[5], base = 10)

βSim = vcat( [parse( Float64, ARGS[6])],
             rand( Uniform(0,1.0), maxJ-1))


K = maxJ
L = 1

nSumers = 25

Γ = zeros(K,K)
for l in 1:L
    Γ[l,l] = parse( Float64, ARGS[7])
end



ηSim = [.5, .3]

numInst = 2

## Standard diffuse priors.
priors = PriorBLP( 50.0, 0.5,
                   zeros(K), diagm( ones(K)*.1),
                   2.0, .1, 0.0, 1.0,
                   zeros(numInst), diagm( ones(numInst)*.1),
                   zeros(2),.01, 3.0)



lamMap = vcat( [ convert( Vector{Int64}, (10*(i-1)+1):(10*i)) for i in 1:div(T,10)])

arrivalDist = []

λ = ones(T)

## We have a flag for what distribution to use, if NegativeBinomial,
## then the next args will be 1,2 so we use split and then parse it.
if ARGS[8] == "true"
    params = parse.( Float64, split( ARGS[9], ",") )
    arrivalDist = repeat( [NegativeBinomial(params[1], params[2])], T)
    λ = ones(T)*log(mean(NegativeBinomial(params[1], params[2])))
else
    ## If we're just a poisson, the rate is just ARGS[9]
    λ = ones(T)*log(parse( Float64, ARGS[9]))
    arrivalDist = Poisson.( exp.(λ))
end





i = ARGS[1]

## Variance on xi is fairly small for the FONC spec, was much larger
## for reduced-form pricing schemes, when this is too large we can see
## crazy price distributions.
draws = rand(Normal(0.0,.1),maxJ*T)
ζ = rand(Normal(),nSumers,K)

## Build the actual data.
data, S, δ  = BuildDataFONC( T, DiscreteUniform(minJ,maxJ), K, L, Γ, βSim, ηSim,
                             draws, numInst, repeat( [Uniform(0,1)], numInst),
                             Multinomial( 1, ones(maxJ)/maxJ ),
                             arrivalDist, ζ, 0.0);

## Only the first 3 searchParameters aren't deprecated.
searchParameters = SearchParameters( parse( Float64, ARGS[10]),
                                     parse( Float64, ARGS[11]),
                                     parse( Float64, ARGS[12]),
                                     .5, 0.0, 0.12, 0.12);

@save "../SimDir/$(ARGS[2])$(i).jld2" data S δ ζ searchParameters priors lamMap βSim ηSim Γ λ





