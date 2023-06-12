import h5py
import numpy as np
import pandas as pd
import pyblp
import sys
import os 
import pathlib

###########################
### CONSTANT PARAMETERS ###
###########################

# pulled from user input
simNum   = int(sys.argv[1])   # x \in [1,100]
numProds = int(sys.argv[2])   # 25, 45, 3
lamSpec  = sys.argv[3]        # sml, big, over, high

# EXAMPLE: simNum = 63; numProds = 3; lamSpec = "high"

# fixed arrivals number to calculate shares
if lamSpec == "high":
    fixedA = 200
else: 
    fixedA = 80                 

# number of MC draws to use in BLP 
ns     = 1000               

# muting output from pyblp 
pyblp.options.verbose = False

######################
### MAIN FUNCTIONS ###
######################

def pullSimData(simNum, numProds, lamSpec):
    '''
    Function reads in the data structures from the jld2 file generated by the Bayes-Julia scripts. 

    Inputs: 
        simNum   -> the simulation number, an integer between [1,100]
        numProds -> must be in the set \{ 25, 45, 3 \}, specifies the number of products 
        dataSpec -> must be in the set \{ sml, big, over, high \}, specifies the arrivals process portion of the DGP

    Note: This is a nasty process. JLD2 are just nested HDF5 files were many of the objects are references. If the data structure is to change on the julia side this will break. Last updated 10/12/2021. 

    DANGEROUS: Reminder that julia is column major while python is row major. This means that some of the matrices are read into python transposed. Be mindful of this when adjusting code. 
    '''
    # make data spec string (now panel data)
    fileName = '../SimDir/J{}/FONCPrice_{}{}J{}.jld2'.format(numProds,lamSpec,numProds,simNum)
    # read in serialized object
    f      = h5py.File(fileName, "r")
    # product chars, note first col denotes price
    X      = np.vstack([f[f[f['data'][()][0]][i]][:].T for i in range(f[f['data'][()][0]].shape[0])])
    # instruments
    Z      = np.vstack([f[f[f['data'][()][2]][i]][:].T for i in range(f[f['data'][()][2]].shape[0])])
    # number of products in a given market 
    nJ     = f[f['data'][()][3]][()].T
    # q for a product-market tuple 
    q      = np.hstack([f[f['data'][()][1]][()].T[i, 0:nJ[i]] for i in range(f[f['data'][()][1]].shape[1])])[:,None]
    # product and market ids 
    idJ    = np.hstack([np.arange(nJ[i]) for i in range(nJ.shape[0])])[:,None]
    idT    = np.hstack([[i]*nJ[i] for i in range(nJ.shape[0])])[:,None]
    # searches for a given market 
    A      = np.hstack([[f[f['data'][()][5]][()][i]]*nJ[i] for i in range(nJ.shape[0])])[:,None]
    # get true shares 
    shares = np.hstack([f['S'][()].T[i,0:nJ[i]] for i in range(nJ.shape[0])])[:,None]
    return X, Z, q, idJ, idT, A, shares

def mkPyBLPdf(X, Z, q, idJ, idT, A, shares):
    '''
    Function takes in numpy objects and makes pandas df with all the correctly named columns to be run in pyblp.

    Inputs: 
        X       -> matrix with product characteristic, first column is assumed to be price 
        Z       -> matrix of instruments 
        q       -> vector of the quantity of purchases 
        idJ     -> product identifiers 
        idT     -> market identifiers 
        A       -> vector of arrivals for a market (should be repeated value for all prods in a mkt)
        shares  -> vector of the true market level purchase probability for a product-market tuple
    '''
    df = pd.DataFrame(np.hstack([idT, idJ, idJ, shares, q, A, X[:,0][:,None]]))
    df.columns = ['market_ids', 'product_ids', 'firm_ids','shares_true', 'Q', 'A', 'prices']  
    # add in product chars 
    for i in range(1, X.shape[1]):
        df['x{}'.format(i)] = X[:,i]
    # add in instruments 
    c = 0
    for i in range(Z.shape[1]):
        df['demand_instruments{}'.format(c)] = Z[:,i]
        df['demand_instruments{}'.format(c+1)] = Z[:,i]**2
        df['cost_shifter{}'.format(i)] = Z[:,i]
        c=c+2
    return df
    
def runPyBLP(df, shareAdj, M, agg):
    '''
    Function takes in a prepped dataframe and specification strings and uses pyBLP to estimate a RC logit model of demand.

    Inputs: 
            df          -> prepped dataframe, made using mkPyBLPdf
            shareAdj    -> specifies how to handel 0's in the empirical shares, 
                            can take on the following values {"adj", "drop"}
                            * adj   - replaces 0 shares with laplace shares (GLS)
                            * drop  - removes observations with empirical shares
                                        equal to 0
            M           -> specifies which market size to use in calculating 
                            empirical shares {"fixed", "observed"}
                            * fixed - uses an arbitrary large value 
                            * observed - uses the drawn arrivals 
            agg         -> specifies if there will be aggregation across 10 markets
                            * agg - aggregate across 10 market groups
    '''    
    ### FIRST CALCULATE AND MAKE ADJUSTMENTS TO EMPIRICAL SHARES
    ### --- SELECT M --- ###
    if M == "fixed":
        # assume that M is a large value
        df['M'] = fixedA
    elif M == "observed":
        # first drop markets with no arrivals 
        df = df.loc[df.A > 0]
        # use observed arrivals 
        df['M'] = df.A.values
    ### --- AGGREGATE --- ###
    if agg == "agg":
        # aggregate markets to groups of 10 (think groupby sum at weekly level)
        df['AgMkt'] = df.market_ids % 50
        dicZ = {z:"mean" for z in [z for z in df.columns.tolist() if "demand_instruments" in z]}
        dicX = {x:"mean" for x in [x for x in df.columns.tolist() if "x" in x]}
        dicAll = {"Q":"sum", "M":"sum", "prices":"mean"}
        dicAll.update(dicZ)
        dicAll.update(dicX)
        df = df.groupby(['AgMkt', 'product_ids']).agg(dicAll).reset_index()
        # make agMkt the new market ids 
        df['market_ids'] = df.AgMkt
        # due to product variability in offerings, need to adjust A 
        df['M'] = df.groupby('market_ids')['M'].transform('max')
    ### --- CALC EMPIRICAL SHARES --- ###
    # if the market size is less then total purchases, set it to the total purchases plus 1 
    df['Q_mkt'] = df.groupby(['market_ids'])['Q'].transform("sum")
    df.loc[df.M <= df.Q_mkt, 'M'] = df.loc[df.M <= df.Q_mkt, 'Q_mkt'] + 1
    # calculate shares given market size def
    df['shares'] = df.Q/df.M
    ### --- SHARE ADJUSTMENT --- ###
    if shareAdj == "adj":
        # replace shares with leplace transformation 
        df['shares'] = (df.M*df.shares + 1)/(df.M + df.groupby('market_ids')['product_ids'].transform('nunique') + 1)
    elif shareAdj == "drop":
        # drop shares above 1 and equal to 0
        df = df[(df.shares > 0) & (df.shares < 1)]
    ### ESTIMATE MODEL WITH GIVEN DATA SPEC WITH PYBLP
    # isolate prod chars 
    prod_str = [x for x in df.columns.tolist() if "x" in x]
    # specify linear params 
    X1_string = '1 + prices + {}'.format(" + ".join(prod_str))
    X1_formulation = pyblp.Formulation(X1_string)
    # specify nonlinear params 
    X2_string = '0 + prices'
    X2_formulation = pyblp.Formulation(X2_string)
    # formulate problem 
    formulation = (X1_formulation, X2_formulation)
    integration = pyblp.Integration('halton', size = ns)
    optimization =  pyblp.Optimization('knitro')
    # initiate problem 
    problem = pyblp.Problem(formulation, df, integration = integration)
    # solve problem 
    results = problem.solve(sigma=0.5, sigma_bounds = (0,5), optimization = optimization, initial_update=True, method='1s')
    # use optimal instruments in 2nd stage 
    instrument_results = results.compute_optimal_instruments(method='approximate')
    # sometimes simulation goes off in a wacky place, so only use optimal instruments if there is variation in them
    if instrument_results.demand_instruments.std() == 0:
        return results
    else: 
        updated_problem = instrument_results.to_problem()
        updated_results = updated_problem.solve(sigma=0.25, sigma_bounds = (0,5), optimization = optimization, method='1s')
        return updated_results 

############
### MAIN ###
############

if __name__ == "__main__":
    # pull data from jld2 files
    X, Z, q, idJ, idT, A, shares = pullSimData(simNum, numProds, lamSpec)
    # make df for pyBLP
    df                  = mkPyBLPdf(X, Z, q, idJ, idT, A, shares)
    # run various specs 
    resultsAdjObsAgg    = runPyBLP(df, "adj", "observed", "agg")
    resultsAdjObsDAgg   = runPyBLP(df, "adj", "observed", "disagg")
    resultsAdjFixAgg    = runPyBLP(df, "adj", "fixed", "agg")
    resultsAdjFixDAgg   = runPyBLP(df, "adj", "fixed", "disagg")
    resultsDrpObsAgg    = runPyBLP(df, "drop", "observed", "agg")
    resultsDrpObsDAgg   = runPyBLP(df, "drop", "observed", "disagg")
    resultsDrpFixAgg    = runPyBLP(df, "drop", "fixed", "agg")
    resultsDrpFixDAgg   = runPyBLP(df, "drop", "fixed", "disagg")
    # put the models all in one ls for ease to construct output 
    models = [resultsAdjObsAgg, resultsAdjObsDAgg, resultsAdjFixAgg, resultsAdjFixDAgg, resultsDrpObsAgg, resultsDrpObsDAgg, resultsDrpFixAgg, resultsDrpFixDAgg]
    # construct results dataframe 
    dfOut            = pd.DataFrame()
    dfOut['method']  = ['adjObsAgg', 'adjObsDAgg', 'adjFixAgg', 'adjFixDAgg', 'dropObsAgg', 'dropObsDAgg', 'dropFixAgg', 'dropFixDAgg'] 
    dfOut['alpha']   = [model.beta[1,0] for model in models]
    dfOut['gamma11'] = [model.sigma[0,0] for model in models]
    for i in range(2,X.shape[1]):
        dfOut['beta{}'.format(i)] = [model.beta[i,0] for model in models]
    dfOut['converged'] = [model.converged for model in models]
    # make sure sub dir is there, if not make one 
    if not os.path.exists(pathlib.Path("../OutputDir/FONC/blpJ{}/".format(numProds))):
        os.mkdir(pathlib.Path("../OutputDir/FONC/blpJ{}/".format(numProds)))
    # save results out 
    fnameOut = "../OutputDir/FONC/blpJ{}/results_blp_{}Sim{}.csv".format(numProds, lamSpec, simNum)
    dfOut.to_csv(fnameOut, index = False)


