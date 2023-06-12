from __future__ import division, print_function
from os import path, linesep, uname, getcwd
from traceback import format_exc
import argparse
import time
import sys
import pathlib
import h5py
import os
import numpy as np
import pandas as pd

from utils.randcoef.rclogit import rclogit
from datetime import datetime

###########################
### CONSTANT PARAMETERS ###
###########################

# pulled from user input
simNum   = int(sys.argv[1])   # x \in [1,100]
numProds = int(sys.argv[2])   # 25, 45
lamSpec  = sys.argv[3]        # sml, big, over 

# taken as given 
fixedA = 80                 # fixed arrivals number to calculate shares
ns     = 1000               # number of simulants 

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
    Function takes in numpy objects and makes pandas df with all the correctly named columns to be run in 
        pyblp.

    Inputs: 
        X       -> matrix with product characterstic, first column is assumed to be price 
        Z       -> matrix of instruments 
        q       -> vector of the quantity of purchases 
        idT     -> market identifiers 
        idJ     -> product identifiers 
        A       -> vector of arrivals for a market (should be repeated value for all prods in a mkt)
        shares  -> vector of the true market level purchase proability for a product-market tuple
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
 
def runGandhi(df, M, agg):
    '''
    Function takes in a prepped dataframe and specification strings and uses Gandhi bounds method to estimate params. 

    Inputs: 
            df          -> prepped dataframe, made using mkPyBLPdf
            M           -> specifies which market size to use in calculating 
                            empirical shares {"fixed", "observed"}
                            * fixed - uses an arbitrary large value 
                            * observed - uses the drawn arrivals 
            agg         -> specifies if there will be aggregation across 10 markets
                            * agg - aggregate across 10 market groups
    '''    
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
    ### ESTIMATE MODEL WITH GIVEN DATA SPEC ###
    # make df of market id and market sizes 
    dfMkt = df[['market_ids','M']].drop_duplicates().reset_index()
    # construct empty dictionary for keyword args in GLS function
    kwargsLogit                  = {}
    # data frame 
    kwargsLogit["df"]            = df
    # name of the column in df that corresponds to shares
    kwargsLogit["shares"]        = "shares"
    # name of columns to be used in linear component of indirect utility 
    kwargsLogit["xLinear"]       = ['prices'] + [x for x in df.columns if 'x' in x]
    # name of columns to be used in the nonlinear component of indirect utility  
    kwargsLogit["xNonLinear"]    = ['prices']
    # complete set of instruments (non-price X and Zs)
    kwargsLogit["instruments"]   = [x for x in df.columns if 'x' in x] + [z for z in df.columns if 'demand_instruments' in z]
    # name of the column in df that corresponds to market ids 
    kwargsLogit["market"]        = ['market_ids']
    # dataframe that has [market_ids, M] (market ids and market size)
    kwargsLogit["dfMarketSizes"] = dfMkt
    # market size variable name 
    kwargsLogit["marketSizes"]   = "M"
    # total number of simulants 
    kwargsLogit["R"]             = ns
    # estimate a la GLS  
    results = rclogit(debugLevel = 3, estimator = "bounds", **kwargsLogit)
    # return just alpha, betas, and gammas 
    return results

###########
### MAIN ###
############

if __name__ == "__main__":
    # pull data from jld2 files
    X, Z, q, idJ, idT, A, shares = pullSimData(simNum, numProds, lamSpec)
    # make df for pyBLP
    df             = mkPyBLPdf(X, Z, q, idJ, idT, A, shares)
    # run various specs 
    resultsObsAgg   = runGandhi(df, "observed", "agg")
    resultsObsDAgg  = runGandhi(df, "observed", "disagg")
    resultsFixAgg   = runGandhi(df, "fixed", "agg")
    resultsFixDAgg  = runGandhi(df, "fixed", "disagg")
    # put the models all in one ls for ease to construct output 
    models = [resultsObsAgg, resultsObsDAgg, resultsFixAgg, resultsFixDAgg]
    # construct output data frame
    dfOut = pd.DataFrame()
    dfOut['method']  = ['ObsAgg', 'ObsDAgg', 'FixAgg', 'FixDAgg'] 
    dfOut['alpha']   = [model.beta[0] for model in models]
    dfOut['gamma11'] = [model.sigma[0] for model in models]
    for i in range(1,X.shape[1]):
        dfOut['beta{}'.format(i)] = [model.beta[i] for model in models]
    # if outdir does not exsit make it 
    if not os.path.exists(pathlib.Path("../OutputDir/FONC/glsJ{}/".format(numProds))):
        os.mkdir(pathlib.Path("../OutputDir/FONC/glsJ{}/".format(numProds)))
    # save out
    fnameOut = "../OutputDir/FONC/glsJ{}/results_gls_{}Sim{}.csv".format(numProds, lamSpec, simNum)
    dfOut.to_csv(fnameOut, index = False)