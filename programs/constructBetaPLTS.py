import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# Plot settings 
csfont              = {'fontname':"Liberation Serif", 'fontsize':18}
palette             = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
plt.rcParams.update({'font.size': 18})

#################
### FUNCTIONS ###
#################

def mkBetaBiasPlots(numProds, lamSpec):
    '''
    Function produces density plots of bias for the linear coefficents. 

    Input: numProds -> Number of products in the DGP 
           lamSpec  -> Size of the arrivals process for the simulated DGP  
    '''
    lsBLP = [0]*100
    for i in range(1,101):
        # pull true beta2 
        fileName   = 'SimDir/J{}/FONCPrice_{}{}J{}.jld2'.format(numProds,lamSpec,numProds,i)
        f          = h5py.File("../" + fileName, "r")
        beta2_true = f["βSim"][()][1]
        # get blp sims
        fileName   = "../OutputDir/FONC/blpJ{}/results_blp_{}Sim{}.csv".format(numProds, lamSpec, i)
        dfBLP      = pd.read_csv(fileName)
        dfBLP['beta2_true'] = beta2_true
        lsBLP[i-1]   = dfBLP
    # concat and construct BLP bias 
    dfBLP = pd.concat(lsBLP)
    dfBLP['beta2_bias'] = dfBLP.beta2 - dfBLP.beta2_true
    # load in PRC bias
    dfPRC = pd.read_csv("../OutputDir/FONC/csv/J{}/FONCPrice_{}{}J_Beta1_Hist.csv".format(numProds, lamSpec, numProds))
    # KDE PLOT 
    fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
    sns.kdeplot(dfPRC.iloc[:,0], label = "PRC",  ls="-", color=palette[3], lw=3)
    sns.kdeplot(dfBLP.beta2_bias, label = "BLP", ls="--", color=palette[0], lw=3)
    L  = plt.legend(loc = "upper left", fontsize = 14.5)
    plt.xlim([-2.0,2.0]) 
    plt.axvline(x = dfPRC.iloc[:,0].mean(),  lw = 2, color = palette[3], ls = ":")
    plt.axvline(x = dfBLP.beta2_bias.mean(), lw = 2, color = palette[0], ls = ":")
    plt.xlabel("Bias",    **csfont, fontproperties=prop) 
    plt.ylabel("Density", **csfont,fontproperties=prop) 
    plt.yticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    plt.xticks(fontsize = 18, family='Liberation Serif', fontproperties=prop)
    ax.tick_params(axis='both', labelsize=18)
    # plt.show()
    plotName = "../figures/mc_beta2Bias_{}Lam_J{}.pdf".format(lamSpec, numProds)
    plt.savefig(plotName,bbox_inches='tight',format= "pdf",dpi=600)
    return None 

############
### MAIN ###
############

if __name__ == "__main__":
    mkBetaBiasPlots(25, "big")
    mkBetaBiasPlots(25, "sml")
    mkBetaBiasPlots(45, "big")

