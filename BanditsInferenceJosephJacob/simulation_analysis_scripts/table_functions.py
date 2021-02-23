import matplotlib
matplotlib.use('Agg')
import pickle
import os
import ipdb
import statsmodels.stats.power as smp
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(1, '../louie_experiments/')
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



def compute_wald_se(df, use_ipw = False):
    mean_1 = df["mean_1"]
    mean_2 = df["mean_2"]
    se = df["wald_type_stat"]*(1/(mean_1 - mean_2))

    if use_ipw == True:
        mean_1 = df["mean_1_ipw"]
        mean_2 = df["mean_2_ipw"]

        se = df["wald_type_stat_ipw"]*(1/(mean_1 - mean_2))
    se = mean_1*(1-mean_1)/df['sample_size_1'] + mean_2*(1-mean_2)/df['sample_size_2']
    se = np.sqrt(np.array(se, dtype = float))
#    se = 1/se

    return se

def save_percentile_table(diffs, abs_diffs, aps,n, alg_key):

    table_save_dir = "../simulation_analysis_saves/Tables/Percentile_tables/n={}/".format(n)
    Path(table_save_dir).mkdir(parents=True, exist_ok=True)
    #one row per ap, diff 
    #one col for 50, 75% perc
    df_dict = {}
    df_dict["50% percentile"] = []
    df_dict["75% percentile"] = []

#    ipdb.set_trace()
    diff_50 = np.percentile(diffs, 50)
    df_dict["50% percentile"].append(diff_50)

    diff_75 = np.percentile(diffs, 75)
    df_dict["75% percentile"].append(diff_75)

    abs_diff_50 = np.percentile(abs_diffs, 50)
    df_dict["50% percentile"].append(abs_diff_50)

    abs_diff_75 = np.percentile(abs_diffs, 75)
    df_dict["75% percentile"].append(abs_diff_75)

    ap_50 = np.percentile(aps, 50)
    df_dict["50% percentile"].append(ap_50)

    ap_75 = np.percentile(aps, 75)
    df_dict["75% percentile"].append(ap_75)

    save_dir = table_save_dir + "/{}.csv".format(alg_key)
    index = ["Difference in Posterior Means", "Absolute Difference in Posterior Means", "Assignment Probability"]
#    cols = ["50% percentile", ""]

#    ipdb.set_trace()
    table_df = pd.DataFrame(df_dict, index = index) 
    table_df = table_df.round(3)
    table_df.to_csv(save_dir)

        
def get_prop_majority_col(df_list):
    prop1_list = []

#   if len(df_list == 0): #intiialzie 0 ES
       #load es 0 dir..
    
   # ipdb.set_trace()
    for df in df_list:
        num_replications = len(df)
        num_steps = df["num_steps"].iloc[0]
        sample_size_1 = np.mean(df["sample_size_1"])
        prop_1 = sample_size_1/num_steps

        prop1_list.append(prop_1)


    prop1_table_col = prop1_list
    prop1_table_col = np.round(np.array(prop1_table_col), 3)
    prop1_table_col_se = np.round(np.sqrt(prop1_table_col*(1-prop1_table_col)/num_replications), 3)
    prop1_table_col_final = ["{} ({})".format(prop1_table_col[i], prop1_table_col_se[i]) for i in range(len(prop1_table_col))]

    return prop1_table_col_final

def get_alg_reward_col(df_list):
    means_list = []
    se_list = []

    for df in df_list:
        num_replications = len(df)
        num_steps = df["num_steps"].iloc[0]
       # if num_steps == 785:
       #     ipdb.set_trace()
        mean_curr = np.mean(df["total_reward"]/num_steps)
        se_curr = np.std(df["total_reward"]/num_steps)/np.sqrt(num_replications)

        means_list.append(mean_curr)
        se_list.append(se_curr)


    reward_table_col = means_list
    reward_table_col = np.round(np.array(reward_table_col), 3)
    #reward_table_col_se = np.round(np.sqrt(reward_table_col*(1-reward_table_col)/num_replications), 3)#This isnt right, reward is mena reward, which is continuous
    reward_table_col_se = np.round(np.array(se_list) ,3)
    reward_table_col_final = ["{} ({})".format(reward_table_col[i], reward_table_col_se[i]) for i in range(len(reward_table_col))]
    return reward_table_col_final

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

    assert num_sims >0
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

def summary_table(df = None, p1 = 0.5, p2 = 0.5, n = None, summary_dict = None, alg_key = None, use_ipw = False):
    #data = {k: [] for k in ["key1","key2","key3"]}   
    if alg_key == "TS":
        use_ipw = True

    df = df[df["num_steps"] == n]
    df = df[:5000]
#    ipdb.set_trace()
    se = compute_wald_se(df) 

    se_var = se.var()
    se_mean = se.mean()

    se_mean_1_ipw = np.nan
    se_mean_2_ipw = np.nan
    se_ipw_wald = np.nan 
    mean_1_ipw = np.nan
    mean_2_ipw = np.nan 

    diff_ipw = np.nan
    abs_diff_ipw = np.nan

    if use_ipw == True:
        se_ipw_wald = compute_wald_se(df, use_ipw = True) 
        se_ipw_wald_var = se_ipw_wald.var()
        se_ipw_wald_mean = se_ipw_wald.mean()
        se_ipw_wald= se_ipw_wald.mean()

        wald_stat_ipw = df['wald_type_stat_ipw']

        wald_stat_97pt5_ipw = np.percentile(wald_stat_ipw, 97.5) 
        wald_stat_2pt5_ipw = np.percentile(wald_stat_ipw, 2.5) 

        mean_1_ipw = df["mean_1_ipw"]
        mean_2_ipw = df["mean_2_ipw"]
        diff_ipw = (mean_1_ipw - mean_2_ipw).mean()
        abs_diff_ipw = (np.abs(mean_1_ipw - mean_2_ipw)).mean()

#        ipdb.set_trace()
        se_mean_1_ipw = (mean_1_ipw*(1-mean_1_ipw)/df["sample_size_1"])**(1/2)
        se_mean_2_ipw = (mean_2_ipw*(1-mean_2_ipw)/df["sample_size_2"])**(1/2)
       # se_mean_1_ipw = np.sqrt(np.array(mean_1_ipw*(1-mean_1_ipw)/df["sample_size_1"], dtype = float))
       # se_mean_2_ipw = np.sqrt(np.array(mean_2_ipw*(1-mean_2_ipw)/df["sample_size_2"], dtype = float))

        se_mean_1_ipw = se_mean_1_ipw.mean()
        se_mean_2_ipw = se_mean_2_ipw.mean()

        mean_1_ipw = mean_1_ipw.mean()
        mean_2_ipw = mean_2_ipw.mean()

    #    summary_dict["mean_1_se_ipw_avg MLE"].append(se_mean_1_ipw.mean())
    #    summary_dict["mean_2_se_ipw_avg MLE"].append(se_mean_2_ipw.mean())

    summary_dict["mean_1_se_avg IPW"].append(se_mean_1_ipw)
    summary_dict["mean_2_se_avg IPW"].append(se_mean_2_ipw)

    wald_stat = df['wald_type_stat']
    se_wald = compute_wald_se(df, use_ipw = False) 
    summary_dict["wald_se_avg MLE"].append(se_wald.mean())
    summary_dict["wald_se_avg IPW"].append(se_ipw_wald)

    wald_stat_97pt5 = np.percentile(wald_stat, 97.5) 
    wald_stat_2pt5 = np.percentile(wald_stat, 2.5) 


    mean_1 = df["mean_1"]
    mean_2 = df["mean_2"]

    diff = mean_1 - mean_2
    diff = diff.mean()
    #ipdb.set_trace()

    abs_diff = np.abs(mean_1 - mean_2)
    abs_diff = np.mean(abs_diff)

    bias_1 = mean_1.mean() - p1 
    bias_2 = mean_2.mean() - p2 

    se_mean_1 = (mean_1*(1-mean_1)/df["sample_size_1"])**(1/2)
    se_mean_2 = (mean_2*(1-mean_2)/df["sample_size_2"])**(1/2)

    summary_dict["mean_1_avg MLE"].append(mean_1.mean())
    summary_dict["mean_2_avg MLE"].append(mean_2.mean())

    summary_dict["mean_1_avg IPW"].append(mean_1_ipw)
    summary_dict["mean_2_avg IPW"].append(mean_2_ipw)

    summary_dict["mean_1_se_avg MLE"].append(se_mean_1.mean())
    summary_dict["mean_2_se_avg MLE"].append(se_mean_2.mean())

    summary_dict["diff MLE"].append(diff)
    summary_dict["abs_diff MLE"].append(abs_diff)
    summary_dict["diff IPW"].append(diff_ipw)
    summary_dict["abs_diff IPW"].append(abs_diff_ipw)

    mean_1_se_emp = np.std(df["mean_1"])/np.sqrt(5000)
    mean_2_se_emp = np.std(df["mean_2"])/np.sqrt(5000)

    summary_dict["mean_1_se_emp"].append(mean_1_se_emp)
    summary_dict["mean_2_se_emp"].append(mean_2_se_emp)


    #data = {k: [] for k in ["key1","key2","key3"]}   


def set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = "Proportions", iseffect = "NoEffect", include_stderr = "NoStderr", upper_bound = 1.0, lower_bound = 0.0, num_sims = 5000):
    '''
    Loop over algs, one for each row
    '''
    table_dict = {}
#    num_sims = len(df_alg_list[0])
    for df_alg, df_alg_key in zip(df_alg_list, df_alg_key_list):
        if len(df_alg) == 0:
            ipdb.set_trace()
        next_row = set_bins(df = df_alg, outcome = outcome, include_stderr = include_stderr, upper_bound = upper_bound, lower_bound = lower_bound)
        table_dict[df_alg_key], col_header = next_row

    table_df = pd.DataFrame(table_dict) 
    table_df.index = col_header 
    table_df = table_df.T

    save_dir = "../simulation_analysis_saves/Tables/{}/{}/{}/num_sims={}/".format(outcome, iseffect, include_stderr, num_sims)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = save_dir + "{}_n={}_numsims={}.csv".format(outcome, num_steps, num_sims)

    table_df.to_csv(save_file) 

def summary_table_helper(df_ts = None, df_unif = None, df_eg0pt1 = None, effect_size = 0, n = None, num_sims = None):

    p1 = 0.5 + effect_size/2
    p2 = 0.5 - effect_size/2
    summary_dict = {k: [] for k in ["mean_1_avg MLE","mean_2_avg MLE", "mean_1_avg IPW","mean_2_avg IPW","mean_1_se_avg MLE", "mean_2_se_avg MLE", "mean_1_se_avg IPW", "mean_2_se_avg IPW", "wald_se_avg MLE", "wald_se_avg IPW", "diff MLE", "abs_diff MLE", "diff IPW", "abs_diff IPW", "mean_1_se_emp", "mean_2_se_emp"]}   

    summary_table(df_ts, p1 = p1, p2 = p2, n = n, summary_dict = summary_dict, alg_key = "TS", use_ipw = True)
    summary_table(df_unif, p1 = p1, p2 = p2, n = n, summary_dict = summary_dict, alg_key = "UR")
    summary_table(df_eg0pt1, p1 = p1, p2 = p2, n = n, summary_dict = summary_dict, alg_key = "EG")

    df = pd.DataFrame.from_dict(summary_dict)
    df = df.round(3)
    #Must respect ordering
    df["Algorithm"] = ["TS", "UR", "EG 0.1"]

    save_dir = "../simulation_analysis_saves/Tables/Summary/EffectSize={}/n={}/num_sims={}/".format(effect_size, n, num_sims)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = save_dir + "n={}_numsims={}.csv".format(n, num_sims)

    df.set_index("Algorithm", inplace = True)
    df.to_csv(save_file)

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
        df_alg_key_list = ["Uniform", "Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"]

        if n == 657/4:
            df_alg_list = [df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_eg0pt1]
            df_alg_key_list = ["Uniform", "Thompson Sampling", "Epsilon Greedy 0.1"]

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





