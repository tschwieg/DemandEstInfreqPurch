Replication code:
"Demand Estimation with Infrequent Purchases and Small Market Sizes"
Ali Hortaçsu, University of Chicago and NBER
Olivia R. Natan, University of California, Berkeley
Hayden Parsley, University of Texas, Austin
Timothy Schwieg, University of Chicago, Booth
Kevin R. Williams, Yale School of Management and NBER
This version: June 12, 2023


# Description

Julia code for replicating the Monte-Carlo excercises in Demand Estimation with Infrequent Purchases and Small Market Sizes. This code simulates markets which price based on a simultaneous move Nash-Betrand Equilibrium with small number of arrivals, leading to high numbers of zeros in purchases. The code then estimates preferences using a Hybrid-Gibbs Sampler MCMC estimator. Alternative estimator optimization uses Artleys Knitro version 12.3.0. A license can be obtained from https://www.artelys.com/solvers/knitro/. Alternatively, SciPy or a custom, user routine may be used instead.


# License

The material is made available through the Quantitative Economics web page as supplementary material. Users are licensed to download, copy, and modify the code. When doing so such users must acknowledge all authors as the original creators and Quantitative Economics as the original publishers. In practice, this means that anyone using the material held within the replication package zip must (i) cite the paper; (ii) cite the replication package, both in the manuscript and in the README of the replication package; (iii) include a Data Availability statement in package to explain how data was obtained (and give proper attribution); and (iv) include the data files themselves in the package.


# Requirements

UNIX or UNIX-like Operatating system.

Tested in Julia 1.6.1 on Linux servers (64-bit)
Package versions:
CSV v0.6.2
CodecZlib v0.7.0
Contour v0.5.7
DataFrames v0.21.8
Distributions v0.25.10
FixedEffectModels v1.6.2
ForwardDiff v0.10.18
GLM v1.5.1
GZip v0.5.1
JLD2 v0.4.3
JuMP v0.21.8
KNITRO v0.10.0
KernelDensity v0.6.3
Latexify v0.15.18
Optim v1.7.3
Plots v1.18.2
PyPlot v2.9.0
Query v1.0.0
SpecialFunctions v1.6.2
StatsBase v0.33.8
StatsPlots v0.14.30
Tables v1.4.4
TranscodingStreams v0.9.5

python:
2:42
knitro==12.3.0
zipp==3.6.0
yarl==1.6.3
wrapt==1.12.1
wheel==0.37.0
Werkzeug==2.0.1
urllib3==1.26.7
typing-extensions==3.10.0.2
tornado==6.1
threadpoolctl==3.0.0
termcolor==1.1.0
tensorflow==2.4.1
tensorflow-estimator==2.6.0
tensorboard==2.4.0
tensorboard-plugin-wit==1.6.0
statsmodels==0.12.2
six==1.16.0
sip==4.19.13
setuptools==58.0.4
seaborn==0.11.2
scipy==1.7.1
scikit-learn==0.24.2
rsa==4.7.2
requests==2.26.0
requests-oauthlib==1.3.0
PyYAML==5.4.1
pytz==2021.1
python-dateutil==2.8.2
PySocks==1.7.1
pyparsing==3.0.4
pyOpenSSL==20.0.1
PyJWT==2.1.0
pycparser==2.20
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyarrow==3.0.0
protobuf==3.17.2
property-cached==1.6.4
pip==21.2.4
Pillow==8.4.0
patsy==0.5.2
pandas==1.3.3
packaging==21.3
opt-einsum==3.3.0
olefile==0.46
oauthlib==3.1.1
numpy==1.20.3
numexpr==2.7.3
mypy-extensions==0.4.3
munkres==1.1.4
multidict==5.1.0
mkl-service==2.4.0
mkl-random==1.2.2
mkl-fft==1.3.0
matplotlib==3.5.0
Markdown==3.3.4
linearmodels==4.24
kiwisolver==1.3.1
Keras==2.4.3
Keras-Preprocessing==1.1.2
joblib==1.1.0
jaxlib==0.1.71+cuda111
jax==0.2.21
importlib-metadata==4.8.1
idna==3.2
h5py==2.10.0
grpcio==1.36.1
google-pasta==0.2.0
google-auth==1.33.0
google-auth-oauthlib==0.4.4
gast==0.4.0
fonttools==4.25.0
flatbuffers==2.0
Cython==0.29.24
cycler==0.11.0
cryptography==3.4.8
coverage==5.5
click==8.0.1
charset-normalizer==2.0.4
chardet==4.0.0
cffi==1.14.6
certifi==2021.10.8
cachetools==4.2.2
brotlipy==0.7.0
Bottleneck==1.3.2
blinker==1.4
attrs==21.2.0
async-timeout==3.0.1
astunparse==1.6.3
astor==0.8.1
aiohttp==3.7.4.post0
absl-py==0.13.0

KNITRO version 12.3.0

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
          9. SearchDistParameter; If UseNegBinom == "false", then this is a single float64 that gives the poisson arrival rate.
                                  If UseNegBinom == "true" then this is a string that contains two float64 values separated by a comma that gives the parameters for the negative binomial distribution. i.e. "25.0,.5"
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

- programs/simBLP.py: Given a simulated data set, the program estimates demand using BLP with different ad-hoc methods to handle zeros in empirical market shares 
    ARGS: 1: simNum; The index of the simulation being run.
          2: numProds; Total number of products in simulated data.
          3: lamSpec; Market size of simulated data. Note: (numProds, lamSpec)               
             tuple specifies which simulated data set to use. 

- programs/simGLS.py: Given a simulated data set, the program estimates demand using the method of Gandhi, Lu, Shi (2023) with different adjustments. 
    ARGS: 1: simNum; The index of the simulation being run.
          2: numProds; Total number of products in simulated data.
          3: lamSpec; Market size of simulated data. Note: (numProds, lamSpec)               
             tuple specifies which simulated data set to use.

- programs/constructTBLS.py: Returns the final tables used in the paper. There are no command line
                             arguments for this file.     


# Replication instructions

Here we describe how to replicate the estimates and figures reported in the paper.

Ensure the working directory is DemandEstInfreqPurch/programs/

Much of the code can be run in parallel, we chose to do so using a
schedule manager, Slurm. It is possible to do this with other
scheduling software, or out of parallel using a for loop in your shell
language of choice. In all places in the replication, simply replace
$((0 + ${SLURM_ARRAY_TASK_ID})) with the appropriate index. In our
case, these calls are placed in individual slurm batch files, using
the following flags.

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-100


0. Ensuring the Correct File structure exists

sh EnsureFileStructure.sh

1. Simulating Data
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J25/FONCPrice_big25J 500 15 25 -2.0 .2 false 25.0 1.875 0.0 0.12817
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J45/FONCPrice_big45J 500 40 45 -2.0 .2 false 25.0 0.855 0.0 0.064575
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J25/FONCPrice_sml25J 500 15 25 -2.0 .2 false 5.0 1.5 0.0 0.13125
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J25/FONCPrice_over25J 500 15 25 -2.0 .2 true 25.0,.5 1.5 0.0 0.13125
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J25/FONCPrice_sin25J 500 15 25 -2.0 .2 false 25.0 .9 0.0 0.10382
julia SimFONC.jl $((0 + ${SLURM_ARRAY_TASK_ID})) J3/FONCPrice_high3J 500 3 4 -2.0 .2 false 200.0 0.196875 0.0 0.21875

2. Calling Estimation

julia RunSims.jl ../SimDir/J25/FONCPrice_big25J((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_big25J((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J45/FONCPrice_big45J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J45/FONCPrice_big45J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_sml25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_sml25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_over25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J25/FONCPrice_over25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J25/FONCPrice_sin25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 1 ../OutputDir/FONC/J25/FONCPrice_sin25J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2

julia RunSims.jl ../SimDir/J3/FONCPrice_high3J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2 25 ../OutputDir/FONC/J3/FONCPrice_high3J$((0 + ${SLURM_ARRAY_TASK_ID})).jld2


3. Aggregating Julia Output

julia AggregateMCs.jl

4. Alternative Estimators

python simBLP.py ${SLURM_ARRAY_TASK_ID} 25 big
python simBLP.py ${SLURM_ARRAY_TASK_ID} 45 big
python simBLP.py ${SLURM_ARRAY_TASK_ID} 25 sml
python simBLP.py ${SLURM_ARRAY_TASK_ID} 3 high

python simGLS.py ${SLURM_ARRAY_TASK_ID} 25 big
python simGLS.py ${SLURM_ARRAY_TASK_ID} 45 big
python simGLS.py ${SLURM_ARRAY_TASK_ID} 25 sml
python simGLS.py ${SLURM_ARRAY_TASK_ID} 3 high

5. Aggregating all output

python constructTBLS.py
