import scipy.special as sc
import pandas as pd 
import ipdb
from pathlib import Path
from scipy import stats
import numpy as np
from scipy.stats import ttest_ind_from_stats
from scipy.special import betaln
BF_CUTOFF = 10**0
BF_CUTOFF = 3.0
BF_CUTOFF = 1.0
BF_CUTOFF = 1/3 #this is acutaly bf 3, since switchign the direction of inequality based on Anna/ Nina discussion
BF_CUTOFF = 1/(2.5) # ditto for this
BF_CUTOFF = 1/(5.0) # ditto for this
BF_CUTOFF = 2.5 # ditto for this




def compute_wald_se(df, use_ipw = False):
    mean_1 = df["mean_1"]
    mean_2 = df["mean_2"]

    if use_ipw == True:
        mean_1 = df["mean_1_ipw"]
        mean_2 = df["mean_2_ipw"]

    se = df["wald_type_stat"]*(1/(mean_1 - mean_2))
    se = 1/se

    return se


def get_all_hyp_tests_for_algo(df_for_num_steps, use_ipw = True, cutoffs = None, es = 0, n = None):

    num_replications = len(df_for_num_steps)
    if use_ipw == True:
        t1_ipw = np.sum(df_for_num_steps["wald_pval_ipw"] < 0.05)/ num_replications
    else:
        t1_ipw = np.nan

    sample_size_1 = df_for_num_steps["sample_size_1"]
    sucesses_1 = df_for_num_steps["mean_1"]*sample_size_1

    sample_size_2 = df_for_num_steps["sample_size_2"]
    sucesses_2 = df_for_num_steps["mean_2"]*sample_size_2

    bf = BF(sucesses_1, sample_size_1, sucesses_2, sample_size_2)
    mean_bf = np.mean(bf)
    var_bf = np.var(bf)
    table_dict = {}
    table_dict["mean"] = mean_bf
    table_dict["variance"] = var_bf
#    columns = [""]

    table_save_dir = "../simulation_analysis_saves/Tables/BayesFactor/EffectSize={}/".format(es)
    Path(table_save_dir).mkdir(parents=True, exist_ok=True)
    table_save_file = table_save_dir + "/n={}.csv".format(n)
#    ipdb.set_trace()

    index = ["Bayes Factor"]
    table_df = pd.DataFrame(table_dict, index = index) 
    table_df = table_df.round(3)
    table_df.to_csv(table_save_file)

    rejected_bf = np.sum(bf < BF_CUTOFF)#reject H0 if BF is lower than 1
    t1_bf = rejected_bf/num_replications


    pvals_welch = welch_test(means_1 = df_for_num_steps["mean_1"], sample_size_1 = df_for_num_steps["sample_size_1"], means_2 = df_for_num_steps["mean_2"], sample_size_2 = df_for_num_steps["sample_size_2"])
    rejected_welch = np.sum(pvals_welch < 0.05)

    t1_welch = rejected_welch/num_replications

    wald_pval = df_for_num_steps['wald_pval'].dropna()
    num_rejected_wald = np.sum(df_for_num_steps['wald_pval'] < .05) #Thompson

    t1_wald = num_rejected_wald / num_replications
    
    t1_simbased_025 = np.nan
    t1_simbased_05 = np.nan

    if cutoffs != None:
        ap = "ap0.25"
        left_cutoff = cutoffs[ap][0]
        right_cutoff = cutoffs[ap][1]

        wald_type_stat = df_for_num_steps['wald_type_stat']
        rejected_wald_simbased = np.sum(np.logical_or(wald_type_stat > right_cutoff,\
                              wald_type_stat < left_cutoff)) #two sided test with empirical cut offs

        t1_simbased_025 = rejected_wald_simbased/num_replications

        ap = "ap0.5"
        left_cutoff = cutoffs[ap][0]
        right_cutoff = cutoffs[ap][1]

        wald_type_stat = df_for_num_steps['wald_type_stat']
        rejected_wald_simbased = np.sum(np.logical_or(wald_type_stat > right_cutoff,\
                              wald_type_stat < left_cutoff)) #two sided test with empirical cut offs
        t1_simbased_05 = rejected_wald_simbased/num_replications


#    std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
#    std_err_t1 = np.round(std_err_t1,3)
#    next_cell += " {} ({})".format(t1_err, std_err_t1)

    tests_list = np.array([t1_wald, t1_welch, t1_bf, t1_ipw, t1_simbased_025, t1_simbased_05])
    tests_list_se = np.round(np.sqrt(tests_list*(1-tests_list)/num_replications), 3)
    tests_list_withse = ["{} ({})".format(tests_list[i], tests_list_se[i]) for i in range(len(tests_list))]

    index = ["Wald Test", "Welch Test", "Bayes Factor (Cutoff {})".format(BF_CUTOFF), "IPW", "Algorithm Induced Test 0.25", "Algorithm Induced Test 0.5"] 

    return index, tests_list_withse 
        
       # test_by_alg_table([t1_ts, t1_welch, t1_ts_bf, t1_ipw])
        
#def test_by_algo_table(tests_per_algo)
#    for 
def BF(sucesses_1, sample_size_1, sucesses_2, sample_size_2):
    """
    Computes Bayes factor for proportions
    """

    a_1 = 1#m is 1, f is 2 in Nina's notes
    a_2 = 1

    b_1 = 1
    b_2 = 1

    R_1 = sucesses_1
    R_2 = sucesses_2

    n_1 = sample_size_1
    n_2 = sample_size_2

    #compute left, middle, right terms
#    left_term = sc.beta(a_1 + a_2 + R_1 + R_2, b_1 + b_2 + n_1 + n_2 - (R_1 + R_2)) / sc.beta(a_1 + a_2, b_1 + b_2)
    logleft_term = sc.betaln(a_1 + a_2 + R_1 + R_2, b_1 + b_2 + n_1 + n_2 - (R_1 + R_2)) - sc.betaln(a_1 + a_2, b_1 + b_2)

 #   middle_term = sc.beta(a_1 + R_1, b_1 + n_1 - R_1) / sc.beta(a_1, b_1)
    logmiddle_term = sc.betaln(a_1 + R_1, b_1 + n_1 - R_1) - sc.betaln(a_1, b_1)

  #  right_term = sc.beta(a_2 + R_2, b_2 + n_2 - R_2) / sc.beta(a_2, b_2)
    logright_term = sc.betaln(a_2 + R_2, b_2 + n_2 - R_2) - sc.betaln(a_2, b_2)

    logBF = logleft_term - (logmiddle_term + logright_term) 
    #BF = left_term / (middle_term * right_term) 
    BF = np.exp(logBF) 

    return BF

def BF_old(sucesses_1, sample_size_1, sucesses_2, sample_size_2):
    """
    Computes Bayes factor for proportions
    may run into Nans since no log
    """

    a_1 = 1#m is 1, f is 2 in Nina's notes
    a_2 = 1

    b_1 = 1
    b_2 = 1

    R_1 = sucesses_1
    R_2 = sucesses_2

    n_1 = sample_size_1
    n_2 = sample_size_2

    #compute left, middle, right terms
    left_term = sc.beta(a_1 + a_2 + R_1 + R_2, b_1 + b_2 + n_1 + n_2 - (R_1 + R_2)) / sc.beta(a_1 + a_2, b_1 + b_2)
    middle_term = sc.beta(a_1 + R_1, b_1 + n_1 - R_1) / sc.beta(a_1, b_1)
    right_term = sc.beta(a_2 + R_2, b_2 + n_2 - R_2) / sc.beta(a_2, b_2)

    BF = left_term / (middle_term * right_term) 

    return BF

def welch_test(means_1, sample_size_1, means_2, sample_size_2):
    
#    ipdb.set_trace()
    num_suc_1 = (means_1*sample_size_1).astype(int)
    num_suc_2 = (means_1*sample_size_1).astype(int)

    num_fail_1 = (sample_size_1 - num_suc_1).astype(int)
    num_fail_2 = (sample_size_1 - num_suc_1).astype(int)

    std1_list = []
    std2_list = []

    for suc, fail in zip(num_suc_1, num_fail_1):
        rewards_1 = np.append(np.ones(suc), np.zeros(fail))
        std_curr = np.std(rewards_1)
        std1_list.append(std_curr)

    for suc, fail in zip(num_suc_2, num_fail_2):
        rewards_2 = np.append(np.ones(suc), np.zeros(fail))
        std_curr = np.std(rewards_2)
        std2_list.append(std_curr)

    std1_list = np.array(std1_list)
    std2_list = np.array(std2_list)

    var1 = means_1*(1 - means_1)
    stds1 = np.sqrt(var1) 

    var2 = means_2*(1 - means_2)
    stds2 = np.sqrt(var2) 

    #testr = ttest_ind_from_stats(mean1 = np.array(means_1), mean2 = np.array(means_2), std1 = np.array(stds1), std2 = np.array(stds2), nobs1 = np.array(sample_size_1), nobs2 = np.array(sample_size_2), equal_var = False)      
    testr = ttest_ind_from_stats(mean1 = np.array(means_1), mean2 = np.array(means_2), std1 = std1_list, std2 = std2_list, nobs1 = np.array(sample_size_1), nobs2 = np.array(sample_size_2), equal_var = False)      

    return testr.pvalue

