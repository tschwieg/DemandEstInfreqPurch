Replication code:
"Demand Estimation with Infrequent Purchases and Small Market Sizes"
Ali Hortaçsu, University of Chicago and NBER
Olivia R. Natan, University of California, Berkeley
Hayden Parsley, University of Texas, Austin
Timothy Schwieg, University of Chicago, Booth
Kevin R. Williams, Yale School of Management and NBER
This version: June 12, 2023


# Description

Julia code for replicating the Monte-Carlo excercises in Demand
Estimation with Infrequent Purchases and Small Market Sizes. This code
simulates markets which price based on a simultaneous move
Nash-Betrand Equilibrium with small number of arrivals, leading to
high numbers of zeros in purchases. The code then estimates
preferences using a Hybrid-Gibbs Sampler MCMC estimator. Alternative
estimator optimization uses Artleys Knitro version 12.3.0. A license
can be obtained from
https://www.artelys.com/solvers/knitro/. Alternatively, SciPy or a
custom, user routine may be used instead.


# License

The material is made available through the Quantitative Economics web
page as supplementary material. Users are licensed to download, copy,
and modify the code. When doing so such users must acknowledge all
authors as the original creators and Quantitative Economics as the
original publishers. In practice, this means that anyone using the
material held within the replication package zip must (i) cite the
paper; (ii) cite the replication package, both in the manuscript and
in the README of the replication package; (iii) include a Data
Availability statement in package to explain how data was obtained
(and give proper attribution); and (iv) include the data files
themselves in the package.


# Requirements

UNIX or UNIX-like Operatating system.

Tested in Julia 1.6.1 on Linux servers (64-bit)
Python 3.8.16
KNITRO version 12.3.0

For Exact package version details please see the end of the file.


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

- programs/PlotBeta.jl: Constructs a csv of mean(beta1) - betaSim1 to be used in plots.
     ARGS: 1: simName; the relative filename of the data to load. For sim i, reads input as "../SimDir/$(simName)$(i).jld2"
           2: nSims; The total number of simulations
           3: estName; Relative filename of the output from estimation. For sim i, reads from: "../OutputDir/FONC/$(estName)$(i).jld2"
           4: outputLocl; Folder for csv output. Outputs to: "../OutputDir/$(outputLoc)/$(simName)_Beta1_Hist.csv"

- programs/


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

6. Generating plots

julia PlotBeta J25/FONCPrice_big25J 100 J25/FONCPrice_big25J FONC/csv/J25/
julia PlotBeta J45/FONCPrice_big45J 100 J45/FONCPrice_big45J FONC/csv/J45/
julia PlotBeta J25/FONCPrice_sml25J 100 J25/FONCPrice_sml25J FONC/csv/J25/
julia PlotBeta J25/FONCPrice_over25J 100 J25/FONCPrice_over25J FONC/csv/J25/
julia PlotBeta J25/FONCPrice_sin25J 100 J25/FONCPrice_sin25J FONC/csv/J25/
julia PlotBeta J3/FONCPrice_high3J 100 J3/FONCPrice_high3J FONC/csv/J3/

python constructBetaPLTS.py
python constructSharePLTS.py

# Package versions:

Julia:
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
_libgcc_mutex 0.1
_openmp_mutex 5.1
abseil-cpp 20211102.0
arrow-cpp 8.0.0
asn1crypto 1.4.0
attrs 22.1.0
blas 1.0
boost-cpp 1.73.0
boto3 1.18.16
botocore 1.21.16
brotli 1.0.9
brotli-bin 1.0.9
bzip2 1.0.8
c-ares 1.18.1
ca-certificates 2022.12.7
cached-property 1.5.2
cairo 1.16.0
certifi 2022.12.7
cffi 1.14.6
cfitsio 3.470
chardet 4.0.0
charset-normalizer 2.0.4
click 8.0.4
click-plugins 1.1.1
cligj 0.7.2
contourpy 1.0.5
cryptography 3.4.7
curl 7.87.0
cycler 0.11.0
dbus 1.13.18
eigen 3.3.7
expat 2.4.9
fastparquet 0.5.0
ffmpeg 4.2.2
fiona 1.8.22
fontconfig 2.14.1
fonttools 4.25.0
freetype 2.12.1
freexl 1.0.6
gdal 3.0.2
geopandas 0.9.0
geopandas-base 0.9.0
geos 3.8.0
geotiff 1.7.0
gflags 2.2.2
giflib 5.2.1
glib 2.69.1
glog 0.5.0
gmp 6.2.1
gnutls 3.6.15
graphite2 1.3.14
grpc-cpp 1.46.1
gst-plugins-base 1.14.0
gstreamer 1.14.0
h5py 3.2.1
harfbuzz 4.3.0
hdf4 4.2.13
hdf5 1.10.6
icu 58.2
idna 3.2
isodate 0.6.0
jmespath 0.10.0
joblib 1.1.1
jpeg 9e
json-c 0.16
kealib 1.4.14
kiltsreader 0.0.1
kiwisolver 1.4.4
krb5 1.19.4
lame 3.100
lcms2 2.12
ld_impl_linux-64 2.38
lerc 3.0
libblas 3.9.0
libboost 1.73.0
libbrotlicommon 1.0.9
libbrotlidec 1.0.9
libbrotlienc 1.0.9
libcblas 3.9.0
libclang 10.0.1
libcurl 7.87.0
libdap4 3.19.1
libdeflate 1.8
libedit 3.1.20221030
libev 4.33
libevent 2.1.12
libffi 3.4.2
libgcc-ng 11.2.0
libgdal 3.0.2
libgfortran-ng 11.2.0
libgfortran5 11.2.0
libgomp 11.2.0
libidn2 2.3.2
libkml 1.3.0
liblapack 3.9.0
libllvm10 10.0.1
libnetcdf 4.8.1
libnghttp2 1.46.0
libopenblas 0.3.20
libopus 1.3.1
libpng 1.6.37
libpq 12.9
libprotobuf 3.20.3
libspatialindex 1.9.3
libspatialite 4.3.0a
libssh2 1.10.0
libstdcxx-ng 11.2.0
libtasn1 4.16.0
libthrift 0.15.0
libtiff 4.5.0
libunistring 0.9.10
libuuid 1.41.5
libvpx 1.7.0
libwebp 1.2.4
libwebp-base 1.2.4
libxcb 1.15
libxkbcommon 1.0.1
libxml2 2.9.14
libxslt 1.1.35
libzip 1.8.0
llvmlite 0.34.0
lz4-c 1.9.4
mapclassify 2.5.0
matplotlib 3.6.2
matplotlib-base 3.6.2
mpmath 1.2.1
msrest 0.6.21
munch 2.5.0
munkres 1.1.4
ncurses 6.4
nettle 3.7.3
networkx 2.8.4
nspr 4.33
nss 3.74
numba 0.51.2
numpy 1.22.3
oauthlib 3.1.1
opencv 4.6.0
opencv-python 4.3.0.36
opencv-python-headless 4.7.0.68
openh264 2.1.1
openjpeg 2.4.0
openssl 1.1.1s
orc 1.7.4
oscrypto 1.2.1
packaging 22.0
pandas 1.2.1
patsy 0.5.3
pcre 8.45
pillow 9.3.0
pip 21.2.4
pixman 0.40.0
ply 3.11
poppler 0.81.0
poppler-data 0.4.11
proj 6.2.1
pyarrow 10.0.1
pyblp 1.0.0
pycparser 2.20
pycryptodomex 3.10.1
pyhdfe 0.1.0
pyjwt 2.1.0
pyopenssl 20.0.1
pyparsing 3.0.9
pyproj 2.6.1.post1
pyqt 5.15.7
pyqt5-sip 12.11.0
python 3.8.16
python-dateutil 2.8.2
python-snappy 0.5.4
python_abi 3.8
pytz 2022.7
qt-main 5.15.2
qt-webengine 5.15.9
qtwebkit 5.212
re2 2022.04.01
readline 8.2
requests 2.26.0
requests-oauthlib 1.3.0
rtree 1.0.1
s3transfer 0.5.0
scikit-learn 1.2.0
scipy 1.8.1
seaborn 0.12.2
setuptools 65.6.3
shapely 1.8.4
sip 6.6.2
six 1.16.0
snappy 1.1.9
sqlite 3.40.1
statsmodels 0.11.1
sympy 1.8
tbb 2021.6.0
threadpoolctl 2.2.0
thrift 0.11.0
tiledb 2.3.3
tk 8.6.12
toml 0.10.2
tornado 6.1
urllib3 1.26.6
utf8proc 2.6.1
wheel 0.37.1
x264 1!157.20191217
xerces-c 3.2.4
xz 5.2.10
zlib 1.2.13
zstd 1.5.2
