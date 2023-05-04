Replication code:
"Demand Estimation with Infrequent Purchases and Small Market Sizes"
Ali Hortaçsu, University of Chicago and NBER
Olivia R. Natan, University of California, Berkeley
Hayden Parsley, University of Texas, Austin
Timothy Schwieg, University of Chicago, Booth
Kevin R. Williams, Yale School of Management and NBER
This version: May 3, 2023


# Description

Julia code for 


# License

The material is made available through the Quantitative Economics web page as supplementary material. Users are licensed to download, copy, and modify the code. When doing so such users must acknowledge all authors as the original creators and Quantitative Economics as the original publishers. In practice, this means that anyone using the material held within the replication package zip must (i) cite the paper; (ii) cite the replication package, both in the manuscript and in the README of the replication package; (iii) include a Data Availability statement in package to explain how data was obtained (and give proper attribution); and (iv) include the data files themselves in the package.


# Requirements

- Julia 1.0 or later version.

Tested in Julia 0.XXXX on Linux servers (64-bit)


# Contents

* Main executables

- programs/SimFONC.jl: simulate data for Monte-Carlo estimation.

    ARGS: 1: index; The index of the simulation being run, i.e. a
             number from 1-(# Simuatlions) typically set by the workload
             manager that parallelizes this call.
          2: simName; The name of the simulation. Files are saved in the following location:
                      "../SimDir/$(ARGS[2])$(i).jld2"
          3: T; Number of markets to simulate. Integer
          4. minJ; minimum number of products in the market. Products are drawn from a DiscreteUniform() distribution with lower bound minJ and upper bound maxJ.
          5: maxJ; maxmimum number of products in the market. See minJ
          6. alpha; the price sensitivity parameter.
          7. Γ; The random coefficient on the price sensitivity parameter.
          8. UseNegBinom; True or False whether or not to use a Negative Binomial distribution for the search distribution.
          9. SearchDistParameter; If UseNegBinom == "false", then this is a single float64 that gives the poisson arrival rate. If UseNegBinom == "true" tehn this is a string that contains two float64 values separated by a comma that gives the parameters for the negative binomial distribution. i.e. "25.0,.5"
          10. searchSigmaDiag; The variance of the candidate distribution used by the Metropolis-Hastings step for the diagonal elements of Gamma.
          11. searchSigmaOffDiag; The variance of the candidate distribution used by the Metropolis-Hastings step for the off-diagonal elements of Gamma.
          12. searchDScale; The variance of the candidate distribution used by the Metropolis-Hastings step for the share draws.
    
- programs/RunSims.jl: Calls estimation
    ARGS: 1: fileName; the relative filename of the data to load for estimation.
          2: maxMixtures; maximum number of clusters allowed by the DP.
          3: outputName; Directory to write output to.
          
- programs/bayes_BLP_DP.jl: Core library function for estimation routine
    This file is never called directly, referenced by RunSims.jl and SimFONC.jl
    
- programs/AggregateMCs.jl: Reads estimation output and produces a
                            table simResults.csv containing key point
                            estimates. There are no command line
                            arguments for this file.

- programs/CheckOutput.jl: Reads the output files created by
                           estimation and produces some coverage
                           output as well as a few diagonostic plots.  
    ARGS: 1: simName; The baseName of the output files: located at "../OutputDir/$(simName)$(index).jld2"
          2: nSims; The total number of simulations
          3: simDirName; The baseName of the simulation files: located at "../SimDir/$(simName)$(index).jld2"
          
- programs/AdjustSearchParameters.jl: Adjusts the searchParameters of
                                      a subset of simulation files.
    ARGS: 1: simName; The baseName of the simulation files: located at "../SimDir/$(simName)$(index).jld2"
          2: nSims; The total number of simulations
          3: gamAdjust; The multiplicative factor by which to adjust the gamma search parameter.
          4: dScaleADjust; The multiplicative factor by which to adjust the share search parameter.
          5: minIndex; The smallest indexed sim to apply this adjustment to.
          6: maxIndex; The largest indexed sim to apply this adjustment to.


# Replication instructions

Here we describe how to replicate the figures reported in the paper. The PNG files referenced below will be generated after running the code.

Ensure the working directory is MCMethods/programs/

## TODO: Resolve Slurm Issues?

0. Ensuring the Correct File structure exists

sh EnsureFileStructure.sh

1. Simulating Data
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_big25J 500 15 25 -2.0 .2 false 25.0 .05 0.0 .175
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_big45J 500 40 45 -2.0 .2 false 25.0 .05 0.0 .175
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_sml25J 500 15 25 -2.0 .2 false 5.0 .05 0.0 .175
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_over25J 500 15 25 -2.0 .2 true 25.0,.5 .05 0.0 .175
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_sin25J 500 15 25 -2.0 .2 false 25.0 .05 0.0 .175
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) FONCPrice_high3J 500 3 3 -2.0 .2 false 25.0 .05 0.0 .175

2. Calling Estimation

julia RunSims.jl ../SimDir/J25/FONCPrice_big25J((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_big25J((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J45/FONCPrice_big45J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J45/FONCPrice_big45J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_sml25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_sml25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_over25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_over25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_sin25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 1 ../OutputDir/FONC/J25/FONCPrice_sin25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2


3. Aggregating Output

julia AggregateMCs.jl
