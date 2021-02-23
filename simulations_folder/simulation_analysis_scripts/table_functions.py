import matplotlib
matplotlib.use('Agg')
import pickle
import os
#import ipdb
import statsmodels.stats.power as smp
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '../../louie_experiments/')
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
import read_config
from output_format import H_ALGO_ACTION_FAILURE, H_ALGO_ACTION_SUCCESS, H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD
from output_format import H_ALGO_ESTIMATED_MU, H_ALGO_ESTIMATED_V, H_ALGO_ESTIMATED_ALPHA, H_ALGO_ESTIMATED_BETA
from output_format import H_ALGO_PROB_BEST_ACTION, H_ALGO_NUM_TRIALS
import beta_bernoulli
#import thompson_policy
from pathlib import Path

EPSILON_PROB = .000001

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

OUTCOME_A = "TableA-Proportions+T1vsDiff" #For Table A
OUTCOME_B = "TableB-T1vsDiff" #For Table A
OUTCOME_C = "TableC-T1vsImba"
OUTCOME_D = "TableD-Proportions+T1vsImba"


def bin_imba_apply_outcome(df = None, upper = 0.5, lower = 0.0, outcome = "Proportions", include_stderr = False):
    """
    percentage
    """
    num_sims = len(df)
    df["imba"] = np.abs(df["sample_size_1"] / (df["sample_size_1"] + df["sample_size_2"]) - 0.5)
    df["wald_reject"] = df["wald_pval"] < 0.05
    bin_curr = df[(lower <= df["imba"]) & (df["imba"] < upper)]
    t1_total = np.round(np.sum(df["wald_reject"])/num_sims, 3)

    if outcome == OUTCOME_D:
        prop = len(bin_curr)/num_sims
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims
#        t1_err = np.round(t1_err, 3)

        if include_stderr == "WithStderr":
            std_err_prop = np.sqrt(prop*(1-prop)/num_sims)
            std_err_prop = np.round(std_err_prop,3)
            next_cell = "{} ({})".format(round(prop,3), std_err_prop)
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell += " {} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{} {}".format(round(prop,3), round(t1_err,3))
    if outcome == OUTCOME_C:
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims      
#        t1_err = np.round(t1_err, 3)
        if include_stderr == "WithStderr":
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell = "{} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{}".format(round(t1_err,3))         

    if outcome == "mean":
        next_cell = np.round(np.mean(df["imba"]), 3)
    if outcome == "std":
        next_cell = np.round(np.std(df["imba"]), 3)
    if outcome == "t1total":
        if include_stderr == "WithStderr":
            next_cell = t1_total 
        else:
            std_err_t1_total = np.sqrt(t1_total*(1-t1_total)/num_sims)
            next_cell = "{} ({})".format(round(t1_total,3), round(std_err_t1_total,3))


    return next_cell

def bin_abs_diff_apply_outcome(df = None, upper = 1.0, lower = 0.0, outcome = "Proportions", include_stderr = False):
    num_sims = len(df)
    df["abs_diff"] = np.abs(df["mean_1"] - df["mean_2"])
    df["wald_reject"] = df["wald_pval"] < 0.05
    bin_curr = df[(lower <= df["abs_diff"]) & (df["abs_diff"] < upper)]
    t1_total = np.round(np.sum(df["wald_reject"])/num_sims, 3)

    if outcome == OUTCOME_A:
        prop = len(bin_curr)/num_sims
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims
#        t1_err = np.round(t1_err, 3)

        if include_stderr == "WithStderr":
            std_err_prop = np.sqrt(prop*(1-prop)/num_sims)
            std_err_prop = np.round(std_err_prop,3)
            next_cell = "{} ({})".format(prop, std_err_prop)
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell += " {} ({})".format(t1_err, std_err_t1)
        else:
            next_cell = "{} {}".format(np.round(prop,3), np.round(t1_err,3))

    if outcome == OUTCOME_B:
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims      
#        t1_err = np.round(t1_err, 3)
        if include_stderr == "WithStderr":
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell = "{} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{}".format(round(t1_err,3))         

    if outcome == "mean":
        next_cell = np.round(np.mean(df["abs_diff"]), 3)
    if outcome == "std":
        next_cell = np.round(np.std(df["abs_diff"]), 3)
    if outcome == "t1total":
        if include_stderr == "WithStderr":
            next_cell = t1_total 
        else:
            std_err_t1_total = np.round(np.sqrt(t1_total*(1-t1_total)/num_sims), 3)
            next_cell = "{} ({})".format(t1_total, std_err_t1_total)

    return next_cell


def set_bins(df = None, lower_bound = 0,  upper_bound = 1.0, step = 0.1, outcome = "Proportions", include_stderr = "NoStderr"):
    '''
    set bins for a row
    '''
    next_row = []
#    ipdb.set_trace()
    bins = np.round(np.arange(lower_bound, upper_bound, step), 3)
    col_header = []
    if outcome.split("-")[0].strip("Table") in "AB":
        mean_cell = bin_abs_diff_apply_outcome(df, outcome = "mean")
        var_cell = bin_abs_diff_apply_outcome(df, outcome = "std")
        t1total_cell = bin_abs_diff_apply_outcome(df, outcome = "t1total")

    elif outcome.split("-")[0].strip("Table") in "CD":
        mean_cell = bin_imba_apply_outcome(df, outcome = "mean")
        var_cell =  bin_imba_apply_outcome(df, outcome = "std")
        t1total_cell = bin_imba_apply_outcome(df, outcome = "t1total")

    next_row.append(mean_cell)
    next_row.append(var_cell)
    next_row.append(t1total_cell)
    col_header.append("Mean")
    col_header.append("Std")
    col_header.append("Type 1 Error Total")

    for lower in bins:
        upper = np.round(lower + step, 3) 
        
        if outcome.split("-")[0].strip("Table") in "AB":
            next_cell = bin_abs_diff_apply_outcome(df, upper, lower, outcome = outcome, include_stderr = include_stderr)
        
            col_header.append("[{}, {})".format(lower, upper))
        elif outcome.split("-")[0].strip("Table") in "CD":
            next_cell = bin_imba_apply_outcome(df, upper, lower, outcome = outcome, include_stderr = include_stderr)

            col_header.append("[{} %, {} %)".format(round(100*lower,2), round(100*upper,2)))#percentage
        next_row.append(next_cell)
#        col_header.append("[{}, {})".format(lower, upper))

    if outcome.split("-")[0].strip("Table") in "CD":
        next_cell = bin_imba_apply_outcome(df, 0.51, 0.48, outcome = outcome, include_stderr = include_stderr)
        next_row.append(next_cell)
        col_header.append("[48 %, 50 %]")
    return next_row, col_header

def set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = "Proportions", iseffect = "NoEffect", include_stderr = "NoStderr", upper_bound = 1.0, lower_bound = 0.0, num_sims = 5000):
    '''
    Loop over algs, one for each row
    '''
    table_dict = {}
#    num_sims = len(df_alg_list[0])
    for df_alg, df_alg_key in zip(df_alg_list, df_alg_key_list):
        next_row = set_bins(df = df_alg, outcome = outcome, include_stderr = include_stderr, upper_bound = upper_bound, lower_bound = lower_bound)
        table_dict[df_alg_key], col_header = next_row

    table_df = pd.DataFrame(table_dict) 
    table_df.index = col_header 
    table_df = table_df.T

    save_dir = "../simulation_analysis_saves/Tables/{}/{}/{}/num_sims={}/".format(outcome, iseffect, include_stderr, num_sims)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = save_dir + "{}_n={}_numsims={}.csv".format(outcome, num_steps, num_sims)

    table_df.to_csv(save_file) 


def table_means_diff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, \
                     title = None, iseffect = "NoEffect", num_sims = 5000):
    '''
    ''' 

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

        df_alg_list = [df_for_num_steps_ts, df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_unif]
        df_alg_list = [df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3]
        df_alg_key_list = ["Uniform", "Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"]

        include_stderr  = "WithStderr"

        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_A, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_B, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_C, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_D, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 


        include_stderr  = "NoStderr"

        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_A, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_B, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_C, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_D, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 



#    fig.suptitle(title)
#    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#      # if not os.path.isdir("plots"):
#      #    os.path.mkdir("plots")
#    save_str_ne = "diff_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "diff_hist/Effect/{}.png".format(title) 
#    if "No Effect" in title:
#	    print("saving to ", save_str_ne)
#	    fig.savefig(save_str_ne)
#    elif "With Effect" in title:
#	    print("saving to ", save_str_e)
#	    fig.savefig(save_str_e)
#
#      #plt.show()
#    plt.clf()
#    plt.close()





