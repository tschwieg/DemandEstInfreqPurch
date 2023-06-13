import pandas as pd
from glob import glob
import numpy as np

pathPRC      = "../OutputDir/simResults.csv"
pathOut      = "../figures/"

alpha_true   = -2.0
gamma_true   = 0.2 

#################
### FUNCTIONS ###
#################

def mkSS(simMethod, df):
  '''
   Function takes in data frame of all simulated results for BLP and GLS estimators, and constructs values reported in tables; Median Absolute Deviation (MAD), Mean Absolute Error (MAE), Bias, and Mean Squared Errors (MSE)
       Inputs: 
                simMethod -> Selected string to indicate which method to handle zeros to    
                                report 
                df        -> Data frame with all simulation results from BLP or GLS estimators 
       Outputs: 
                a         -> list of reported metrics for alpha (price term)
                g11       -> list of reported metrics for gamma[1,1] (price random coef)
  '''
  # make storage objects for lists 
  a = [0]*6
  g11 = [0]*6
  ### --- ALPHA --- ###
  # MAD
  a[0] = np.abs((df.loc[df.method == simMethod, 'alpha'] - alpha_true)).quantile(.5).round(2)
  # MAE 
  a[1] = np.abs(df.loc[df.method == simMethod, 'alpha'] - alpha_true).mean().round(2)
  # BIAS 
  a[2] = (df.loc[df.method == simMethod, 'alpha'] - alpha_true).mean().round(2)
  # MSE 
  a[3] = ((df.loc[df.method == simMethod, 'alpha'] - alpha_true)**2).mean().round(2)
  # 2.5th percentile of bias terms across simulations
  a[4] = (df.loc[df.method == simMethod, 'alpha'] - alpha_true).quantile(.025).round(2)
  # 97.5th percentile of bias terms across simulations
  a[5] = (df.loc[df.method == simMethod, 'alpha'] - alpha_true).quantile(.975).round(2)
  ### --- GAMMA11 --- ###
  # MAD
  g11[0] = np.abs((df.loc[df.method == simMethod, 'gamma11'] - gamma_true)).quantile(.5).round(2)
  # MAE 
  g11[1] = np.abs((df.loc[df.method == simMethod, 'gamma11'] - gamma_true)).mean().round(2)
  # BIAS 
  g11[2] = (df.loc[df.method == simMethod, 'gamma11'] - gamma_true).mean().round(2)
  # MSE 
  g11[3] = ((df.loc[df.method == simMethod, 'gamma11'] - gamma_true)**2).mean().round(2)
  # 2.5th percentile of bias terms across simulations 
  g11[4] = (df.loc[df.method == simMethod, 'gamma11'] - gamma_true).quantile(.025).round(2)
  # 97.5th percentile of bias terms across simulations
  g11[5] = (df.loc[df.method == simMethod, 'gamma11'] - gamma_true).quantile(.975).round(2)
  return a, g11

def mkTbl(numProds, lamSpec):
  '''
  Function takes in arguments about the DGP and constructs tables of summary stats of performance for each estimator and the ad-hoc adjustments. Output is the tex tables using f strings. 
  
  Inputs: 
         numProds -> number of products relating to the simulated data DGP 
         lamSpec  -> arrival specification 
  '''
  ### --- Pull BLP Estimates --- ###
  fnamesOut = "../OutputDir/FONC/blpJ{}/results_blp_{}Sim*.csv".format(numProds, lamSpec)
  dfls      = [pd.read_csv(x) for x in glob(fnamesOut)]
  df        = pd.concat(dfls)
  # limit to alpha and gamma to converged sims  
  df = df.loc[df.converged == True, ['method', 'alpha', 'gamma11']]
  df = df[np.abs(df.alpha) < 40].reset_index(drop = True)
  df = df[df.gamma11 < 20].reset_index(drop = True)
  # string formating for tables 
  fs = lambda ls : [["\\ "*(i>0)*(i<4) + "{:.2f}".format(y[i]) for i in range(len(y))] for y in ls]
  # make summary stats for outcomes of BLP models 
  dropFixDAgg  = fs(mkSS("dropFixDAgg", df))
  adjFixDAgg   = fs(mkSS("adjFixDAgg", df))
  dropFixAgg   = fs(mkSS("dropFixAgg", df))
  adjFixAgg    = fs(mkSS("adjFixAgg", df))
  dropObsDAgg  = fs(mkSS("dropObsDAgg", df))
  adjObsDAgg   = fs(mkSS("adjObsDAgg", df))
  dropObsAgg   = fs(mkSS("dropObsAgg", df))
  adjObsAgg    = fs(mkSS("adjObsAgg", df))
  BLP          = [dropFixDAgg, adjFixDAgg, dropFixAgg, adjFixAgg, dropObsDAgg, adjObsDAgg, dropObsAgg, adjObsAgg]
  ### --- Pull GLS Estimates --- ###
  fnamesOut = "../OutputDir/FONC/glsJ{}/results_gls_{}Sim*.csv".format(numProds, lamSpec)
  dfGls      = [pd.read_csv(x) for x in glob(fnamesOut)]
  dfG        = pd.concat(dfGls)
  dfG        = dfG[['method', 'alpha', 'gamma11']]
  # make summary stats for GLS results 
  fixDAgg    = fs(mkSS("FixDAgg", dfG))
  fixAgg     = fs(mkSS("FixAgg",  dfG))
  obsDAgg    = fs(mkSS("ObsDAgg", dfG))
  obsAgg     = fs(mkSS("ObsAgg",  dfG))
  GLS        = [fixDAgg, fixAgg, obsDAgg, obsAgg]
  ### --- Pull PRC Estimates --- ###
  dfPRC = pd.read_csv(pathPRC)
  dfPRC = dfPRC.iloc[[0,4],:].reset_index(drop = True)
  if (numProds == 25) & (lamSpec == "sml"):
      colsPRC = ['Median Abs Bias - Small_Lam_25J', 'Mean Absolute Deviation - Small_Lam_25J', 'Mean - Small_Lam_25J', 'Variance - Small_Lam_25J', '2.5Pct. - Small_Lam_25J', '97.5Pct. - Small_Lam_25J', ]
  if (numProds == 25) & (lamSpec == "big"):
      colsPRC = ['Median Abs Bias - 25J', 'Mean Absolute Deviation - 25J', 'Mean - 25J', 'Variance - 25J', '2.5Pct. - 25J', '97.5Pct. - 25J']
  if (numProds == 45) & (lamSpec == "big"):
      colsPRC = ['Median Abs Bias - 45J', 'Mean Absolute Deviation - 45J', 'Mean - 45J', 'Variance - 45J', '2.5Pct. - 45J', '97.5Pct. - 45J']
  dfPRC = dfPRC[colsPRC].reset_index(drop = True)
  # clean strings up 
  dfPRC = pd.DataFrame([dfPRC[x].str.replace("$", "").str.replace("\\", "").astype(float) for x in dfPRC.columns]).T
  PRC = fs(dfPRC.values.tolist())
  ### ------ Make Table ----- ###
  # make row names
  row_leads      = [0]*26 
  row_leads[0]   = "Poisson-RC & $-$          & $-$          & $-$    &"
  row_leads[1]   = "           &              &              &        &"
  row_leads[2]   = "BLP (1995) & 80           & No           & Drop   &"
  row_leads[3]   = "           &              &              &        &"
  row_leads[4]   = "BLP (1995) & 80           & No           & Adjust &"
  row_leads[5]   = "           &              &              &        &"    
  row_leads[6]   = "BLP (1995) & 80           & Yes          & Drop   &" 
  row_leads[7]   = "           &              &              &        &"
  row_leads[8]   = "BLP (1995) & 80           & Yes          & Adjust &" 
  row_leads[9]   = "           &              &              &        &"
  row_leads[10]  = "BLP (1995) & Realized     & No           & Drop   &" 
  row_leads[11]  = "           &              &              &        &"
  row_leads[12]  = "BLP (1995) & Realized     & No           & Adjust &" 
  row_leads[13]  = "           &              &              &        &" 
  row_leads[14]  = "BLP (1995) & Realized     & Yes          & Drop   &" 
  row_leads[15]  = "           &              &              &        &"
  row_leads[16]  = "BLP (1995) & Realized     & Yes          & Adjust &" 
  row_leads[17]  = "           &              &              &        &"
  row_leads[18]  = "GLS (2023) & 80           & No           & $-$    &" 
  row_leads[19]  = "           &              &              &        &"
  row_leads[20]  = "GLS (2023) & 80           & Yes          & $-$    &" 
  row_leads[21]  = "           &              &              &        &"
  row_leads[22]  = "GLS (2023) & Realized     & No           & $-$    &"  
  row_leads[23]  = "           &              &              &        &"
  row_leads[24]  = "GLS (2023) & Realized     & Yes          & $-$    &" 
  row_leads[25]  = "           &              &              &        &"
  # declare header of table 
  tbl_ss = f'''
      \\setlength\\tabcolsep{{2.5pt}}
      \\begin{{tabular}}{{cccccccccccc}}
          \\toprule
          
              \\multicolumn{{4}}{{c}}{{}} & \\multicolumn{{4}}{{c}}{{$\\alpha$}} & \\multicolumn{{4}}{{c}}{{$\\Gamma$}} \\\\
              \\cline{{5-12}} 
              Estimator & M  & Aggregate  & Zeros & MAD & MAE & Bias & MSE & MAD & MAE & Bias & MSE \\\\
              \\midrule 
  '''
  # declare footer of table 
  tbl_footer = f'''
  \\bottomrule
  \\end{{tabular}}
  '''
  # construct row counter 
  c = 0 
  # first make PRC row 
  tbl_row  = row_leads[c] + " & ".join(PRC[0][0:4]) + " & "+ " & ".join(PRC[1][0:4])  + "\\\\" + '\n'
  c = c + 1
  # make second PRC row with percentiles 
  tbl_row2 = row_leads[c] + " & ".join([" "]*3) + "({},{}) &".format(PRC[0][4], PRC[0][5]) + " & ".join([" "]*4) + "({},{}) ".format(PRC[1][4], PRC[1][5]) + "\\\\" + ' \n'
  tbl_ss   = tbl_ss + tbl_row + tbl_row2 
  c        = c + 1
  # now iterate through BLP estimators 
  for i in range(len(BLP)):
    # first make row with means/medians  
    tbl_row  = row_leads[c] + " & ".join(BLP[i][0][0:4]) + " & "+ " & ".join(BLP[i][1][0:4])  + "\\\\" + '\n'
    c = c + 1
    # make second row with percentiles 
    tbl_row2 = row_leads[c] + " & ".join([" "]*3) + "({},{}) &".format(BLP[i][0][4], BLP[i][0][5]) + " & ".join([" "]*4) + "({},{}) ".format(BLP[i][1][4], BLP[i][1][5]) + "\\\\" + ' \n'
    tbl_ss   = tbl_ss + tbl_row + tbl_row2 
    c        = c + 1
  # finally, iterate through GLS estimators 
  for j in range(len(GLS)):
    # first make row with means/medians  
    tbl_row  = row_leads[c] + " & ".join(GLS[j][0][0:4]) + " & "+ " & ".join(GLS[j][1][0:4])  + "\\\\" + '\n'
    c = c + 1
    # make second row with percentiles 
    tbl_row2 = row_leads[c] + " & ".join([" "]*3) + "({},{}) &".format(GLS[j][0][4], GLS[j][0][5]) + " & ".join([" "]*4) + "({},{}) ".format(GLS[j][1][4], GLS[j][1][5]) + "\\\\" + ' \n'
    tbl_ss   = tbl_ss + tbl_row + tbl_row2 
    c        = c + 1
  # append footer
  tbl_ss = tbl_ss + tbl_footer
  return tbl_ss

def mkTbl_O():
    '''
        Function constructs output table for "other" specifications, overdispered and single mixing component, and constructs bias tables for PRC. 
    '''
    # pull poisson-logit RC results 
    dfPRC = pd.read_csv(pathPRC)
    dfPRC = dfPRC.iloc[[0,3],:].reset_index(drop = True)
    dfPRC = dfPRC[[col for col in dfPRC.columns if "Coverage" not in col]]
    dfPRC = dfPRC[[x for x in dfPRC.columns if "Row Label" not in x]]
    dfPRC = pd.DataFrame([dfPRC[x].str.replace("$", "").str.replace("\\", "").astype(float) for x in dfPRC.columns]).T
    fs    = lambda ls : [["\\ "*(x>0) + "{:.2f}".format(x) for x in y] for y in ls]
    NB_cols = ["Mean - Over_25J", "2.5Pct. - Over_25J", "97.5Pct. - Over_25J"]
    MS_cols = ["Mean - Sin25J", "2.5Pct. - Sin25J", "97.5Pct. - Sin25J"]
    PRC_NB = fs(dfPRC[[x for x in dfPRC.columns if "Over" in x]].values.tolist())
    PRC_MS = fs(dfPRC[[x for x in dfPRC.columns if "Sin" in x]].values.tolist())
    ### ------ make table ----- ###
    # declare header of table 
    tbl_ss = f'''
    \\begin{{tabular}}{{lcc}}
    \\toprule
        & $ \\lambda \\sim \\text{{NegBinom}}(25, 0.5)$ & Misspecified Residual \\\\
        \\midrule 
    '''
    # declare footer of table 
    tbl_footer = f'''
    \\bottomrule
    \\end{{tabular}}
    '''
    # names of rows and the rows to put in null values for blp sims 
    var_names = ['$\\alpha$', '$\\Gamma_{{11}}$']
    null_names = ['$\\eta_1$', '$\\eta_2$', '$\\lambda$']
    # put mcs you want in table in a master list
    ls = [PRC_NB, PRC_MS]
    # iterate and append each row
    for i in range(len(var_names)):
        # if var name in variables not recovered in blp sims make a null row
        if var_names[i] in null_names:
            continue
        else:
            # pull sum stats in order of table 
            mean_bias = [ls[j][i][0] for j in range(len(ls))]
            quar_bias = ["({}, {})".format(ls[j][i][1], ("\\" not in ls[j][i][2])*"\\ " + ls[j][i][2]) for j in range(len(ls))]
            # make row string 
            tbl_row = "{} & ".format(var_names[i]) + " & ".join(mean_bias) + "\\\\" + "\n"
            tbl_row2 = " & " + " & ".join(quar_bias) + "\\\\" + "\n"
            # append string to header 
            tbl_ss = tbl_ss + tbl_row + tbl_row2
    # append footer on last step 
    tbl_ss = tbl_ss + tbl_footer
    return tbl_ss

def mkTblJ3():
  '''
  Function makes table for large arrivals and J = 3. Reported in appendix 
  '''
  numProds = 3 
  lamSpec  = 'high'
  ### --- Pull BLP Estimates --- ###
  fnamesOut = "../OutputDir/FONC/blpJ{}/results_blp_{}Sim*.csv".format(numProds, lamSpec)
  dfls      = [pd.read_csv(x) for x in glob(fnamesOut)]
  df        = pd.concat(dfls)
  # limit to alpha and gamma  
  df = df.loc[df.converged == True, ['method', 'alpha', 'gamma11']].reset_index(drop = True)
  df = df[np.abs(df.alpha) < 40].reset_index(drop = True)
  df = df[df.gamma11 < 20].reset_index(drop = True)
  # fix string formatting
  fs  = lambda ls : [["\\ "*(i>0)*(i<4) + "{:.2f}".format(y[i]) for i in range(len(y))] for y in ls]
  # make summary stats for RELATIVE bias of BLP models 
  dropFixDAgg  = fs(mkSS("dropFixDAgg", df))
  adjFixDAgg   = fs(mkSS("adjFixDAgg", df))
  dropFixAgg   = fs(mkSS("dropFixAgg", df))
  adjFixAgg    = fs(mkSS("adjFixAgg", df))
  dropObsDAgg  = fs(mkSS("dropObsDAgg", df))
  adjObsDAgg   = fs(mkSS("adjObsDAgg", df))
  dropObsAgg   = fs(mkSS("dropObsAgg", df))
  adjObsAgg    = fs(mkSS("adjObsAgg", df))
  BLP          = [dropFixDAgg, adjFixDAgg, dropFixAgg, adjFixAgg, dropObsDAgg, adjObsDAgg, dropObsAgg, adjObsAgg]
  ### --- Pull PRC Estimates --- ###
  dfPRC = pd.read_csv(pathPRC)
  dfPRC = dfPRC.iloc[[0,4],:].reset_index(drop = True)
  colsPRC = ['Median Abs Bias - J3', 'Mean Absolute Deviation - J3', 'Mean - J3', 'Variance - J3', '2.5Pct. - J3', '97.5Pct. - J3']
  dfPRC = dfPRC[colsPRC]
  # clean strings up 
  dfPRC = pd.DataFrame([dfPRC[x].str.replace("$", "").str.replace("\\", "").astype(float) for x in dfPRC.columns]).T
  PRC = fs(dfPRC.values.tolist())
  ### ------ Make Table ----- ###
  # make row names
  row_leads      = [0]*18
  row_leads[0]   = "Poisson-RC & $-$          & $-$          & $-$    &"
  row_leads[1]   = "           &              &              &        &"
  row_leads[2]   = "BLP (1995) & 200           & No           & Drop   &"
  row_leads[3]   = "           &              &              &        &"
  row_leads[4]   = "BLP (1995) & 200           & No           & Adjust &"
  row_leads[5]   = "           &              &              &        &"    
  row_leads[6]   = "BLP (1995) & 200           & Yes          & Drop   &" 
  row_leads[7]   = "           &              &              &        &"
  row_leads[8]   = "BLP (1995) & 200           & Yes          & Adjust &" 
  row_leads[9]   = "           &              &              &        &"
  row_leads[10]  = "BLP (1995) & Realized     & No           & Drop   &" 
  row_leads[11]  = "           &              &              &        &"
  row_leads[12]  = "BLP (1995) & Realized     & No           & Adjust &" 
  row_leads[13]  = "           &              &              &        &" 
  row_leads[14]  = "BLP (1995) & Realized     & Yes          & Drop   &" 
  row_leads[15]  = "           &              &              &        &"
  row_leads[16]  = "BLP (1995) & Realized     & Yes          & Adjust &" 
  row_leads[17]  = "           &              &              &        &"
  # declare header of table 
  tbl_ss = f'''
      \\setlength\\tabcolsep{{2.5pt}}
      \\begin{{tabular}}{{cccccccccccc}}
          \\toprule
          
              \\multicolumn{{4}}{{c}}{{}} & \\multicolumn{{4}}{{c}}{{$\\alpha$}} & \\multicolumn{{4}}{{c}}{{$\\Gamma$}} \\\\
              \\cline{{5-12}} 
              Estimator & M  & Aggregate  & Zeros & MAD & MAE & Bias & MSE & MAD & MAE & Bias & MSE \\\\
              \\midrule 
  '''
  # declare footer of table 
  tbl_footer = f'''
  \\bottomrule
  \\end{{tabular}}
  '''
  # construct row counter 
  c = 0 
  # first make PRC row 
  tbl_row  = row_leads[c] + " & ".join(PRC[0][0:4]) + " & "+ " & ".join(PRC[1][0:4])  + "\\\\" + '\n'
  c = c + 1
  # make second PRC row with percentiles 
  tbl_row2 = row_leads[c] + " & ".join([" "]*3) + "({},{}) &".format(PRC[0][4], PRC[0][5]) + " & ".join([" "]*4) + "({},{}) ".format(PRC[1][4], PRC[1][5]) + "\\\\" + ' \n'
  tbl_ss   = tbl_ss + tbl_row + tbl_row2 
  c        = c + 1
  # now iterate through BLP estimators 
  for i in range(len(BLP)):
    # first make row with means/medians  
    tbl_row  = row_leads[c] + " & ".join(BLP[i][0][0:4]) + " & "+ " & ".join(BLP[i][1][0:4])  + "\\\\" + '\n'
    c = c + 1
    # make second row with percentiles 
    tbl_row2 = row_leads[c] + " & ".join([" "]*3) + "({},{}) &".format(BLP[i][0][4], BLP[i][0][5]) + " & ".join([" "]*4) + "({},{}) ".format(BLP[i][1][4], BLP[i][1][5]) + "\\\\" + ' \n'
    tbl_ss   = tbl_ss + tbl_row + tbl_row2 
    c        = c + 1
  # append footer
  tbl_ss = tbl_ss + tbl_footer
  return tbl_ss

############
### MAIN ###
############

if __name__ == "__main__":
    # CONSTRUCT ALL THE TABLES 
    smlLam25_landscape = mkTbl(25, "sml")
    bigLam25_landscape = mkTbl(25, "big")
    bigLam45_landscape = mkTbl(45, "big")
    hugeLam3_landscape = mkTblJ3()
    otherSpecs         = mkTbl_O()
    # SAVE TABLES OUT
    with open(pathOut + "mcBiasTbl_smlLam_J25.tex",'w') as f:
        f.write(smlLam25_landscape)
    with open(pathOut + "mcBiasTbl_bigLam_J25.tex",'w') as f:
        f.write(bigLam25_landscape)
    with open(pathOut + "mcBiasTbl_bigLam_J45.tex",'w') as f:
        f.write(bigLam45_landscape)
    with open(pathOut + "mcBiasTbl_hugeLam_J3.tex",'w') as f:
        f.write(hugeLam3_landscape)
    with open(pathOut + "mcBiasTbl_otherSpecs.tex",'w') as f:
        f.write(otherSpecs)

