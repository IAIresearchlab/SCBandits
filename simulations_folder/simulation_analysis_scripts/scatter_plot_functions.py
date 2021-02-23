import matplotlib
matplotlib.use('Agg')
#matplotlib.use("gtk")
#matplotlib.use('Qt5Agg')

from rectify_vars_and_wald_functions import *
import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(1, '../../louie_experiments/')
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
from pathlib import Path
import glob
import numpy as np
import read_config
from output_format import H_ALGO_ACTION_FAILURE, H_ALGO_ACTION_SUCCESS, H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD
from output_format import H_ALGO_ESTIMATED_MU, H_ALGO_ESTIMATED_V, H_ALGO_ESTIMATED_ALPHA, H_ALGO_ESTIMATED_BETA
from output_format import H_ALGO_PROB_BEST_ACTION, H_ALGO_NUM_TRIALS
import beta_bernoulli
import scipy.stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
#import thompson_policy
import ipdb
EPSILON_PROB = .000001

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_minssratio_vs_algs(ax, df_list, x_label, y_label):
#    ipdb.set_trace()
    idx = 0
    ind = np.arange(4)
    ax.set_xticks(ind)
    labels = ('Uniform', 'EG0pt3', 'EG0pt1', 'TS')
    ax.set_xticklabels(labels)
    for df in df_list:
        df[df[y_label] > 1.0] = 1/(df[df[y_label] > 1.0]) #Ratio is smaller sample size/ larger sample size

        df_reject = df[df[x_label] == True]
        x_idx = np.zeros(len(df_reject[x_label])) + idx
        jitter = np.random.normal(0, 0.1, len(x_idx))/2
        if idx == 0:
            ax.scatter(x_idx + jitter,df_reject[y_label], color = 'red', label = "Rejected Null With Wald Test") 
        else:
            ax.scatter(x_idx + jitter,df_reject[y_label], color = 'red') 

        df_accept = df[df[x_label] == False]
        x_idx = np.zeros(len(df_accept[x_label])) + idx
        jitter = np.random.normal(0, 0.1, len(x_idx))/2

        if idx == 0:
            ax.scatter(x_idx + jitter, df_accept[y_label], color = 'blue', label = "Failed to Reject Null With Wald Test")
        else:
            ax.scatter(x_idx + jitter, df_accept[y_label], color = 'blue') 
        idx +=1


   
def scatter_ratio(df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     to_check_ts = None):
    '''
    
    '''
    if load_df == True:
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)

        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
   # SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2)
    df_eg0pt1 = df_eg0pt1.dropna()
    wald_pval_eg0pt1 = (1 - scipy.stats.norm.cdf(np.abs(df_eg0pt1["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_eg0pt1["Wald Rejected"] = wald_pval_eg0pt1 < 0.05
    df_eg0pt1.to_csv("overview_csvs/EG0pt1/eg0pt1_overview_noNa_n={}.csv".format(n))  

    df_eg0pt3 = df_eg0pt3.dropna()
    wald_pval_eg0pt3 = (1 - scipy.stats.norm.cdf(np.abs(df_eg0pt3["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_eg0pt3["Wald Rejected"] = wald_pval_eg0pt3 < 0.05
    df_eg0pt3.to_csv("overview_csvs/EG0pt3/eg0pt3_overview_noNa_n={}.csv".format(n))  

    df_ts = df_ts.dropna()
    wald_pval_ts = (1 - scipy.stats.norm.cdf(np.abs(df_ts["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_ts["Wald Rejected"] = wald_pval_ts < 0.05
    df_ts.to_csv("overview_csvs/TS/ts_overview_noNa_n={}.csv".format(n))  

    df_unif = df_unif.dropna()
    wald_pval_unif = (1 - scipy.stats.norm.cdf(np.abs(df_unif["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_unif["Wald Rejected"] = wald_pval_unif < 0.05                                                                                      #print(data)
    df_unif.to_csv("overview_csvs/unif/unif_overview_noNa_n={}.csv".format(n))
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

        df_list = [df_for_num_steps_unif, df_for_num_steps_eg0pt3, df_for_num_steps_eg0pt1, df_for_num_steps_ts]
#        df_list = [df_for_num_steps_eg0pt1]
        #df_list = [df_for_num_steps_ts]
#        df_list = [df_for_num_steps_unif]

        y_label = "ratio"
        x_label = "Wald Rejected" 
        plot_minssratio_vs_algs(ax = ax[i], df_list = df_list, x_label = x_label, y_label = y_label)
        num_replications = len(df_for_num_steps_eg0pt1)

#        
        ax[i].set_xlabel("Number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,1.02)
        ax[i].set_ylabel("Minimum Sample Size Ratio \n Min($\\frac{n_1}{n_2}$, $\\frac{n_2}{n_1}$)")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/scatter_ratio_waldreject/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/scatter_ratio_waldreject/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)

    save_str_ne = save_dir_ne + "{}.png".format(title)
    save_str_e = save_dir_e + "{}.png".format(title)

#    save_str_ne = "../simulation_analysis_saves/scatter_ratio_waldreject/NoEffect/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/scatter_ratio_waldreject/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne, bbox_inches = "tight")
    elif "With Effect" in title:
	    print("saving to ", save_str_e, bbox_inches = "tight")
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()

def plot_correlation(fig, ax, df_list, x_label, y_label, num_steps, ax_idx):
#    ipdb.set_trace()
    idx = 0
    df = df_list[0]
   
#    for df in df_list: #This loop not needed

    df_reject = df[df["Wald Rejected"] == True]
    xvals = np.abs(df_reject[x_label]/num_steps - 0.5) #Ratio is smaller sample size/ larger sample size
    yvals = np.abs(df_reject[y_label.format(2)] - df_reject[y_label.format(1)]) #Ratio is smaller sample size/ larger sample size
    if ax_idx == 0:
        ax.scatter(xvals, yvals, color = 'red', label = "Rejected Null With Wald Test") 
    else:
        ax.scatter(xvals,yvals, color = 'red') 

    df_accept = df[df["Wald Rejected"] == False]

    xvals = np.abs(df_accept[x_label]/num_steps - 0.5) #Ratio is smaller sample size/ larger sample size
    yvals = np.abs(df_accept[y_label.format(2)] - df_accept[y_label.format(1)]) #Ratio is smaller sample size/ larger sample size
    proportion_reject = len(df_reject)/len(df)

    yvals_all = np.abs(df[y_label.format(2)] - df[y_label.format(1)]) #Ratio is smaller sample size/ larger sample size
    xvals_all = np.abs(df[x_label]/num_steps - 0.5) #Ratio is smaller sample size/ larger sample size
    proportion_reject = np.round(proportion_reject, 3)
    coeff, p = spearmanr(xvals_all, yvals_all)
    coeff = np.round(coeff, 3)
    p = np.round(p, 3)

    coeff_pear, p_pear = pearsonr(xvals_all, yvals_all)
    coeff_pear = np.round(coeff_pear, 3)
    p_pear = np.round(p_pear, 3)

    if ax_idx == 0:
        ax.scatter(xvals, yvals, color = 'blue', label = "Failed to Reject Null With Wald Test")
        ax.legend(loc = "upper center", bbox_to_anchor = (1.2, 1.276))
    else:
        ax.scatter(xvals,yvals , color = 'blue') 
    ax.text(0.02, 0.75,"Proprtion Rejected (Type 1 Error) = {} \nSpearman's Correlation Coefficent = {} \nwith pvalue = {}\n Pearon's Correlation Coefficent = {} \nwith pvalue = {}".format(proportion_reject, coeff, p, coeff_pear, p_pear))
#    if ax_idx == 0 and 0:
 #      leg1 = ax.legend((p_red[0], p_blue[0]), "Rejected Null Hypothesis With Wald Test", "Failed To Reject Null Hypothesis With Wald Test", bbox_to_anchor = (1.0, 1.076))
      # ax.add_artist(leg1)
     #  handles, labels = ax.get_legend_handles_labels()
    
    #   fig.legend(handles, ["a","g"], loc='upper right', prop={'size': 50})
       



def scatter_correlation_helper_outer(df = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     df_ts = None):

    alg_key_list = ["TS", "EG0pt1", "EG0pt3", "Uniform"]
    alg_name_list = ["Thompson Sampling (TS)","Epsilon Greedy 0.1 (EG0.1)","Epsilon Greedy 0.3 (EG0.3)", "Uniform"]

    for alg_key, alg_name in zip(alg_key_list, alg_name_list):
        title_scatter_corr = "{} ".format(alg_name) + "Difference in arm means (|$\hatp_1$ - $\hatp_2$|) vs. |Proportion of samples in Condtion 1 - 0.5|" + " For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

        scatter_correlation(df_eg0pt1 = df_eg0pt1 , df_eg0pt3 = df_eg0pt3,\
                                     df_unif = df_unif, df_ts = df_ts,\
                                           title = title_scatter_corr, \
                                           n = n, num_sims = num_sims, alg_key = alg_key)
        

def scatter_correlation(df = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     df_ts = None, alg_key = "TS"):
    '''
    maybe something like |proportion condition 1 - 0.5| vs. difference in means? Something which captures the imbalance directly
    
    '''

    df_eg0pt1 = df_eg0pt1
    wald_pval_eg0pt1 = (1 - scipy.stats.norm.cdf(np.abs(df_eg0pt1["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    #df_eg0pt1["Wald Rejected"] = wald_pval_eg0pt1 < 0.05
    df_eg0pt1["Wald Rejected"] = df_eg0pt1["wald_pval"] < 0.05

    #df_eg0pt3 = df_eg0pt3.dropna()
    wald_pval_eg0pt3 = (1 - scipy.stats.norm.cdf(np.abs(df_eg0pt3["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_eg0pt3["Wald Rejected"] = df_eg0pt3["wald_pval"] < 0.05

    #df_ts = df_ts.dropna()
    wald_pval_ts = (1 - scipy.stats.norm.cdf(np.abs(df_ts["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_ts["Wald Rejected"] = df_ts["wald_pval"] < 0.05


#    df_unif = df_unif.dropna()
    wald_pval_unif = (1 - scipy.stats.norm.cdf(np.abs(df_unif["wald_type_stat"].dropna())))*2 #Two sided, symetric, so compare to 0.05
    df_unif["Wald Rejected"] = df_unif["wald_pval"] < 0.05

    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

        #df_list = [df_for_num_steps_unif, df_for_num_steps_eg0pt3, df_for_num_steps_eg0pt1, df_for_num_steps_ts]
      #  df_list = [df_for_num_steps_eg0pt3]
        alg_dict = {"TS":df_for_num_steps_ts, "EG0pt1":df_for_num_steps_eg0pt1, "EG0pt3":df_for_num_steps_eg0pt3, "Uniform":df_for_num_steps_unif}
        df_list = [alg_dict[alg_key]]
#        df_list = [df_for_num_steps_ts]
        #df_list = [df_for_num_steps_ts]
      #  df_list = [df_for_num_steps_unif]
       # bins = np.arange(0, 1.01, .025)


        x_label = "sample_size_1"
        y_label = "mean_{}" 
        plot_correlation(fig, ax = ax[i], df_list = df_list, x_label = x_label, y_label = y_label, num_steps = num_steps, ax_idx = i)
        num_replications = len(df_for_num_steps_eg0pt1)

#       
#        
        ax[i].set_xlabel("|Proportion of samples in Condtion 1 - 0.5| For Number of participants = {} = {}".format(size_vars[i], num_steps))
      #  ax[i].legend()
        ax[i].set_ylim(0,1.02)
        ax[i].set_xlim(0, 0.501)
        ax[i].set_ylabel("Difference in Arm Mean Estimates |$\hatp1$ - $\hatp2$|")
        i +=1  
    fig.suptitle(title)
    fig.subplots_adjust(top=0.80)
#    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/scatter_correlation/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/scatter_correlation/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)

    save_str_ne = save_dir_ne + "{}.png".format(title)
    save_str_e = save_dir_e + "{}.png".format(title)

#    save_str_ne = "../simulation_analysis_saves/scatter_correlation/NoEffect/{}/{}.png".format(alg_key, title) 
#    save_str_e = "../simulation_analysis_saves/scatter_correlation/Effect/{}/{}.png".format(alg_key, title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne, bbox_inches = "tight")
    elif "With Effect" in title:
	    print("saving to ", save_str_e, bbox_inches = "tight")
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()
