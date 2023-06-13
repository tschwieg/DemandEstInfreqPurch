import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm

from multiprocessing import Pool
from matplotlib import pyplot as plt

###########################
### CONSTANT PARAMETERS ###
###########################

# taken as given 
fixedA = 80                 # fixed arrivals number to calculate shares

##################
### PLOT STUFF ###
##################

plt.rcParams.update({'font.size': 20})
csfont          = {'fontname':"Liberation Serif", 'fontsize':20}
palette         = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
sns.set(style="white",color_codes=False)

######################
### MAIN FUNCTIONS ###
######################

def pullSimData(simNum, numProds, lamSpec):
    '''
    Function reads in the data structures from the jld2 file generated by the Bayes-Julia scripts. 

    Inputs: 
        simNum   -> the simulation number, an integer between [1,1000]
        dataSpec -> either "smallLam" or "medLam", specifies which data generation process is pulled

    Note: This is a nasty process. JLD2 are just nested HDF5 files were many of the objects are references. If the data structure is to change on the julia side this will break.

    DANGEROUS: Reminder that julia is column major while python is row major. This means that some of the matrices are read into python transposed. Be mindful of this when adjusting code. 
    '''
    # make data spec string (now panel data)
    fileName = 'SimDir/J{}/FONCPrice_{}{}J{}.jld2'.format(numProds,lamSpec,numProds,simNum)
    # read in serialized object
    f      = h5py.File("../" + fileName, "r")
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

def mkPyBLPdf(M, shareAdj, agg, X, Z, q, idJ, idT, A, shares):
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
        dicAll = {"Q":"sum", "M":"sum", "prices":"mean", 'shares_true':"mean"}
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
        # replace the 0's with a small positive number
        # df.loc[df.shares == 0, 'shares'] = adj0
        # replace shares with leplace transformation 
        df['shares'] = (df.M*df.shares + 1)/(df.M + df.groupby('market_ids')['product_ids'].transform('nunique') + 1)
        # for markets with shares > 1, recalculate them using A + 1 as M
        # df.loc[df.shares == 1, 'shares'] = df.loc[df.shares == 1, 'Q']/(df.loc[df.shares == 1, 'M'] + 1)
    elif shareAdj == "drop":
        # drop shares above 1 and equal to 0
        df = df[(df.shares > 0) & (df.shares < 1)]
    return df
   
def mkdfSmallLam(i):
    # pull data from jld2 files
    X, Z, q, idJ, idT, A, shares = pullSimData(i, 25, "sml")
    # construct data frame used in estimation
    df_fix_adj_dagg      = mkPyBLPdf('fixed', 'adj', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_fix_adj_agg       = mkPyBLPdf('fixed', 'adj', 'agg',X, Z, q, idJ, idT, A, shares)
    df_fix_drp_dagg      = mkPyBLPdf('fixed', 'drop', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_fix_drp_agg       = mkPyBLPdf('fixed', 'drop', 'agg',X, Z, q, idJ, idT, A, shares)
    df_obs_adj_dagg      = mkPyBLPdf('observed', 'adj', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_obs_adj_agg       = mkPyBLPdf('observed', 'adj', 'agg',X, Z, q, idJ, idT, A, shares)
    df_obs_drp_dagg      = mkPyBLPdf('observed', 'drop', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_obs_drp_agg       = mkPyBLPdf('observed', 'drop', 'agg',X, Z, q, idJ, idT, A, shares)
    # add simulation indicator 
    df_fix_adj_dagg['sim_id'] = i
    df_fix_adj_agg['sim_id'] = i
    df_fix_drp_dagg['sim_id'] = i
    df_fix_drp_agg['sim_id'] = i
    df_obs_adj_dagg['sim_id'] = i
    df_obs_adj_agg['sim_id'] = i
    df_obs_drp_dagg['sim_id'] = i
    df_obs_drp_agg['sim_id'] = i
    return (df_fix_adj_dagg, df_fix_adj_agg, df_fix_drp_dagg, df_fix_drp_agg, df_obs_adj_dagg, df_obs_adj_agg, df_obs_drp_dagg, df_obs_drp_agg)

def mkdfMedLam(i):
    # pull data from jld2 files
    X, Z, q, idJ, idT, A, shares = pullSimData(i, 25, "big")
    # construct data frame used in estimation
    df_fix_adj_dagg      = mkPyBLPdf('fixed', 'adj', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_fix_adj_agg       = mkPyBLPdf('fixed', 'adj', 'agg',X, Z, q, idJ, idT, A, shares)
    df_fix_drp_dagg      = mkPyBLPdf('fixed', 'drop', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_fix_drp_agg       = mkPyBLPdf('fixed', 'drop', 'agg',X, Z, q, idJ, idT, A, shares)
    df_obs_adj_dagg      = mkPyBLPdf('observed', 'adj', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_obs_adj_agg       = mkPyBLPdf('observed', 'adj', 'agg',X, Z, q, idJ, idT, A, shares)
    df_obs_drp_dagg      = mkPyBLPdf('observed', 'drop', 'disagg',X, Z, q, idJ, idT, A, shares)
    df_obs_drp_agg       = mkPyBLPdf('observed', 'drop', 'agg',X, Z, q, idJ, idT, A, shares)
    # add simulation indicator 
    df_fix_adj_dagg['sim_id'] = i
    df_fix_adj_agg['sim_id'] = i
    df_fix_drp_dagg['sim_id'] = i
    df_fix_drp_agg['sim_id'] = i
    df_obs_adj_dagg['sim_id'] = i
    df_obs_adj_agg['sim_id'] = i
    df_obs_drp_dagg['sim_id'] = i
    df_obs_drp_agg['sim_id'] = i
    return (df_fix_adj_dagg, df_fix_adj_agg, df_fix_drp_dagg, df_fix_drp_agg, df_obs_adj_dagg, df_obs_adj_agg, df_obs_drp_dagg, df_obs_drp_agg)

#############
### MAIN  ###
#############
if __name__ == "__main__":
    # make smallLam simulation dfs 
    p = Pool(64)
    results_SL = p.map(mkdfSmallLam, range(1,101))
    p.close()
    p.join()
    # make medLam simulation dfs 
    p = Pool(64)
    results_ML = p.map(mkdfMedLam, range(1,101))
    p.close()
    p.join()
    # construct data frames from pool outputs
    df_sLam_fix_adj_dagg      = pd.concat([results_SL[i][0] for i in range(len(results_SL))])
    df_sLam_fix_adj_agg       = pd.concat([results_SL[i][1] for i in range(len(results_SL))])
    df_sLam_fix_drp_dagg      = pd.concat([results_SL[i][2] for i in range(len(results_SL))])
    df_sLam_fix_drp_agg       = pd.concat([results_SL[i][3] for i in range(len(results_SL))])
    df_mLam_fix_adj_dagg      = pd.concat([results_ML[i][0] for i in range(len(results_ML))])
    df_mLam_fix_adj_agg       = pd.concat([results_ML[i][1] for i in range(len(results_ML))])
    df_mLam_fix_drp_dagg      = pd.concat([results_ML[i][2] for i in range(len(results_ML))])
    df_mLam_fix_drp_agg       = pd.concat([results_ML[i][3] for i in range(len(results_ML))])
    #### ------ SMALL LAM PLOTS ------ ###
    # share bias
    fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
    sns.kdeplot(df_sLam_fix_drp_dagg.shares_true - df_sLam_fix_drp_dagg.shares, label = "Drop - Disagg.", ls="-", color=palette[0], lw=3)
    sns.kdeplot(df_sLam_fix_drp_agg.shares_true - df_sLam_fix_drp_agg.shares, label = "Drop - Agg.", ls="--", color=palette[1], lw=3)
    sns.kdeplot(df_sLam_fix_adj_dagg.shares_true - df_sLam_fix_adj_dagg.shares, label = "Adjust - Disagg.", ls = "-.", color=palette[4], lw=3)
    sns.kdeplot(df_sLam_fix_adj_agg.shares_true - df_sLam_fix_adj_agg.shares, label = "Adjust - Agg.", ls=":", color=palette[3], lw=3)
    L  = plt.legend(loc = "upper left", fontsize = 14.5)
    plt.xlim([-0.15,0.15]) 
    plt.axvline(x = 0.0, lw = 2, color = palette[2], ls = ":")
    plt.xlabel("True Shares - Empirical Shares", **csfont, fontproperties=prop) 
    plt.ylabel("Density", **csfont,fontproperties=prop) 
    plt.yticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    plt.xticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    ax.tick_params(axis='both', labelsize=18)
    plt.savefig("../figures/mc_shareBias_smallLam_J25.pdf",bbox_inches='tight',format= "pdf",dpi=600)
    #### ------ MED LAM PLOTS ------ ###
    # share bias
    fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
    sns.kdeplot(df_mLam_fix_drp_dagg.shares_true - df_mLam_fix_drp_dagg.shares, label = "Drop - Disagg.", ls="-", color=palette[0], lw=3)
    sns.kdeplot(df_mLam_fix_drp_agg.shares_true - df_mLam_fix_drp_agg.shares, label = "Drop - Agg.", ls="--", color=palette[1], lw=3)
    sns.kdeplot(df_mLam_fix_adj_dagg.shares_true - df_mLam_fix_adj_dagg.shares, label = "Adjust - Disagg.", ls = "-.", color=palette[4], lw=3)
    sns.kdeplot(df_mLam_fix_adj_agg.shares_true - df_mLam_fix_adj_agg.shares, label = "Adjust - Agg.", ls=":", color=palette[3], lw=3)
    L  = plt.legend(loc = "upper left", fontsize = 14.5)
    # plt.setp(L.texts, family='Liberation Serif', fontsize = 20 ,fontproperties=prop) 
    plt.xlim([-0.15,0.15]) 
    plt.axvline(x = 0.0, lw = 2, color = palette[2], ls = ":")
    plt.xlabel("True Shares - Empirical Shares", **csfont, fontproperties=prop) 
    plt.ylabel("Density", **csfont,fontproperties=prop) 
    plt.yticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    plt.xticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    ax.tick_params(axis='both', labelsize=18)
    # plt.show()
    plt.savefig("../figures/mc_shareBias_MedLam_J25.pdf",bbox_inches='tight',format= "pdf",dpi=600)
    #### ------ EPSILON ADJUSTMENT PLOT ------ ###
    df_sLam_0 = pd.concat([df_sLam_fix_adj_dagg, df_sLam_fix_adj_agg])
    df_mLam_0 = pd.concat([df_mLam_fix_adj_dagg, df_mLam_fix_adj_agg])
    fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
    sns.kdeplot(df_sLam_0.loc[df_sLam_0.Q == 0, 'shares_true'], label = r"$\lambda = 5$", ls="-", color=palette[0], lw=3)
    sns.kdeplot(df_mLam_0.loc[df_mLam_0.Q == 0, 'shares_true'], label = r"$\lambda = 25$", ls="--", color=palette[2], lw=3)
    L  = plt.legend(fontsize = 14.5)
    plt.xlabel("True Shares", **csfont, fontproperties=prop) 
    plt.ylabel("Density", **csfont,fontproperties=prop) 
    plt.yticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    plt.xticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    ax.tick_params(axis='both', labelsize=18)
    plt.axvline(x = df_sLam_0.loc[df_sLam_0.Q == 0, 'shares'].mean(), lw = 2, color = palette[2], ls = ":")
    plt.xlim([0.0001,0.15]) 
    plt.savefig("../figures/mc_shareBias_EpsAdjust_J25.pdf",bbox_inches='tight',format= "pdf",dpi=600)