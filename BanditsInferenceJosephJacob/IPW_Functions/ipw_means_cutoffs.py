import matplotlib
#matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
import math
import sys
sys.path.insert(1, '/home/jacobnogas/Sofia_Villar_Fall_2019/banditalgorithms/src/louie_experiments/')
import statsmodels.stats.power as smp
from output_format import H_ALGO_ACTION_FAILURE, H_ALGO_ACTION_SUCCESS, H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD
from output_format import H_ALGO_ESTIMATED_MU, H_ALGO_ESTIMATED_V, H_ALGO_ESTIMATED_ALPHA, H_ALGO_ESTIMATED_BETA
from output_format import H_ALGO_PROB_BEST_ACTION, H_ALGO_NUM_TRIALS
import beta_bernoulli
import thompson_policy
import ng_normal
import run_effect_size_simulations
import run_effect_size_simulations_beta
import read_config
from scipy import stats
#matplotlib.use('QT4Agg')

from get_assistments_rewards import *
reward_file = "../../empirical_data/experiments_data2.csv"
IPW_MEAN_HEADER = "IPWMeanAction{}"
HT_MEAN_HEADER = "HTMeanAction{}"
MLE_MEAN_HEADER = "MLEMeanAction{}"
MAX_WEIGHT_COUNT_HEADER = "CountOfMaxWeights"
SAMPLES_HEADER = "TotalSamples"
MAX_ALLOWED_WEIGHT_HEADER = "MaxWeightParam"
NUM_STEPS_HEADER = "NumSteps"
NUM_SAMPLES_BY_ACTION_HEADER = "NumSamplesAction{}"
PRIOR_MEAN_HEADER = "PriorMean"

IPW_OUT_FILE_SUFFIX = "IPWInfo.pkl" 
IPW_OUT_FILE_CSV_SUFFIX = "IPWInfo.csv" 
WRITE_CSV = False

IPWS_COL = "ipws"
EPSILON_PROB = .000001

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

USE_CACHED_PROBS = True # only relevant for binary rewards - normal rewards never cached because getting the same sequence is prob 0


print(plt.get_backend())

def calculate_all_ipw_estimates(num_sims, step_sizes, outfile_directory, outfile_prefix,
                                 prior, num_samples = 100, num_actions = 2, binary_rewards = True,
                                 config = {}):
    assert num_actions == 2
    read_config.apply_defaults(config)
    cached_probs = {}
    rows = []
    for i in range(num_sims):
        for num_steps in step_sizes:
            if not binary_rewards or not USE_CACHED_PROBS:
                cached_probs = {} # reinitialize the caching every cycle if we aren't caching
                
            if binary_rewards:
                cur_output_file = run_effect_size_simulations_beta.get_output_filename(outfile_directory, num_steps, i)
            else:
                cur_output_file = run_effect_size_simulations.get_output_filename(outfile_directory, num_steps, i)
            #print("processing file:",cur_output_file)
            ipw_row = calculate_ipw(cur_output_file, num_samples, num_actions, cached_probs, prior, binary_rewards, 
                                    config = config)
            #print("processing completed:",cur_output_file)

            ipw_row[NUM_STEPS_HEADER] = num_steps
            rows.append(ipw_row)
    
    dataframe_headers = []
    ipw_headers = [IPW_MEAN_HEADER,HT_MEAN_HEADER,MLE_MEAN_HEADER, NUM_SAMPLES_BY_ACTION_HEADER]
    for header in ipw_headers:
        for action in range(num_actions):
            dataframe_headers.append(header.format(action + 1))
    dataframe_headers.append(MAX_WEIGHT_COUNT_HEADER)
    dataframe_headers.append(NUM_STEPS_HEADER)
    df = pd.DataFrame.from_records(rows, columns = dataframe_headers)
    df[MAX_ALLOWED_WEIGHT_HEADER] = config[read_config.MAX_WEIGHT_HEADER]
    df[SAMPLES_HEADER] = num_samples
#     if binary_rewards:
#         outfile_prefix = run_effect_size_simulations_beta.get_outfile_prefix(outfile_directory, bandit_type_prefix, effect_size, n)
#     else: 
#         outfile_prefix = run_effect_size_simulations.get_outfile_prefix(outfile_directory, bandit_type_prefix, effect_size, n, variance)
    print("writing to", outfile_prefix + IPW_OUT_FILE_SUFFIX) 
    df.to_pickle(outfile_prefix + IPW_OUT_FILE_SUFFIX)
    if WRITE_CSV:
        df.to_csv(outfile_prefix + IPW_OUT_FILE_CSV_SUFFIX)


def create_models_normal(actions_df, prior, num_actions):
    assert num_actions == 2
    cache_keys = [] # no cache_keys for normally-dist. rewards
    all_models = []
    for action in range(num_actions):
        cur_models = [ng_normal.NGNormal(mu=mu, k=k, alpha=alpha, beta=beta) for (mu,k, alpha,beta) in 
                      zip(actions_df.loc[:,H_ALGO_ESTIMATED_MU.format(action + 1)],
                          actions_df.loc[:,H_ALGO_ESTIMATED_V.format(action + 1)],
                          actions_df.loc[:,H_ALGO_ESTIMATED_ALPHA.format(action + 1)],
                          actions_df.loc[:,H_ALGO_ESTIMATED_BETA.format(action + 1)])]
        # add in the one for the prior
        cur_models.insert(0, ng_normal.NGNormal(mu=prior[0], k=prior[1], alpha=prior[2], beta=prior[3]))
        all_models.append(cur_models)
    return all_models, cache_keys

def create_models_binary(actions_df, prior, num_actions):
    assert num_actions == 2

    all_models = []
    cache_keys = [[] for _ in range(actions_df.shape[0])]
    action = 0
    
   # print(actions_df.loc[:,H_ALGO_ACTION_SUCCESS.format(action + 1)])
   # print('Failures------------')
    #print(actions_df.loc[:,H_ALGO_ACTION_FAILURE.format(action + 1)])
    
    for action in range(num_actions):
        
        [cache_keys[i].extend((successes,failures)) for (i,successes,failures) in zip(range(actions_df.shape[0]),actions_df.loc[:,H_ALGO_ACTION_SUCCESS.format(action + 1)],actions_df.loc[:,H_ALGO_ACTION_FAILURE.format(action + 1)])]
#        print((successes, failures)\
#                      for (successes,failures) in\
#                      zip(actions_df.loc[:,H_ALGO_ACTION_SUCCESS.format(action + 1)],\
#                                         actions_df.loc[:,H_ALGO_ACTION_FAILURE.format(action + 1)]))
        
        cur_models = [beta_bernoulli.BetaBern(successes, failures)\
                      for (successes,failures) in\
                      zip(actions_df.loc[:,H_ALGO_ACTION_SUCCESS.format(action + 1)],\
                                         actions_df.loc[:,H_ALGO_ACTION_FAILURE.format(action + 1)])]
        # add in the one for the prior
        cur_models.insert(0, beta_bernoulli.BetaBern(prior[0], prior[1]))
        all_models.append(cur_models)
    # Add in a cache key for the prior
    cache_keys.insert(0, prior*num_actions)
    return all_models,cache_keys
        



def calculate_ipw_means(actions_infile, num_samples, num_actions = 2, cached_probs={}, 
                  prior = [1,1], binary_rewards = True, config = {}):
    assert num_actions == 2
    read_config.apply_defaults(config)
    match_num_samples = config[read_config.MATCH_NUM_SAMPLES_HEADER]
    smoothing_adder = config[read_config.SMOOTHING_ADDER_HEADER]
    max_weight = config[read_config.MAX_WEIGHT_HEADER]
    ipw_row = {}
    ipws = []
    actions_df = pd.read_csv(actions_infile,skiprows=1)
    max_weights = 0
    
   # print(actions_df)
   
   #---------------------------------------------- #Only needed if have not cached
#   
#    if binary_rewards:
#        all_models, cache_keys = create_models_binary(actions_df, prior, num_actions)
#        ipw_row[PRIOR_MEAN_HEADER] = prior[0] / sum(prior)
#
#    else:
#        all_models, cache_keys = create_models_normal(actions_df, prior, num_actions)
#        ipw_row[PRIOR_MEAN_HEADER] = prior[0]
#
#    
#    print("all_models" ,len(all_models)) #list of two (each action) lists. Each sublist contains 1 model for each row
#    saved_probs = [];
#    modified_probs_exist = False
#    cache_key = None
#    for row, i in zip(actions_df.iterrows(), range(len(all_models[0]) - 1)): # we skip the last model because it has the probabilities for the next sample that would be assigned
#        if len(cache_keys) > i:
#            cache_key = tuple(cache_keys[i])
#            
#        # First, look in the file to see if we have a probability we can use - if so, use it
#        if True and not np.isnan(row[1][H_ALGO_PROB_BEST_ACTION.format(1)]) and \
#            (not match_num_samples or num_samples == row[1][H_ALGO_NUM_TRIALS]): # we can use a cached probability
#            probs = [row[1][H_ALGO_PROB_BEST_ACTION.format(1)], row[1][H_ALGO_PROB_BEST_ACTION.format(2)]]
#        elif cache_key is not None and cache_key in cached_probs:
#            # Next, look in the cach keys
#            print("cached probs not found, use calc_ipw_main first")
#            exit()
#            probs = cached_probs[cache_key]
#            modified_probs_exist = True
#        else:
#            print("cached probs not found, use calc_ipw_main first")
#            exit()
#            #Finally, need to calculate probability action 1 is better to use for ipw and then add to cache
#            cur_models = [models[i] for models in all_models] # ith row model, but list; one el for each action
#       #     print("cur_models", len(cur_models))
#            counts = thompson_policy.estimate_probability_condition_assignment(None, num_samples, num_actions, cur_models)
#            print(len(counts))
#            probs = [count / num_samples for count in counts]
#            cached_probs[cache_key] = probs
#            modified_probs_exist = True
#        
#        # Save the unsmoothed probability if asked for and if the probability stored currently isn't present or isn't for the right number of samples
#        #if config[read_config.SAVE_MODIFIED_ACTIONS_FILE_HEADER]:
#        if True:
#            saved_probs.append(probs)
#            
#        # Perform smoothing on the probability vector
#        probs = [(prob * num_samples + smoothing_adder) / (num_samples + smoothing_adder * num_actions) for prob in probs]
#        for j in range(len(probs)):
#            probs[j] = max(probs[j], 0 + EPSILON_PROB)
#            probs[j] = min(probs[j], 1 - EPSILON_PROB)
#        
#        # Determine which probability is relevant based on which condition this sample was assigned to
#        condition_assigned = int(actions_df.iloc[i].loc[H_ALGO_ACTION])
#        prob = probs[condition_assigned - 1] # map back to 0 indexing
#        #print("prob:", prob)
#        weight = 1/prob
#        if weight > max_weight:
#            max_weights += 1
#            weight = max_weight
#        ipws.append(weight)
#        
#    print("actions file:", actions_infile)
##     print("config says to save:",config[read_config.SAVE_MODIFIED_ACTIONS_FILE_HEADER])
#    print("modified probs exist:", modified_probs_exist) #OLD commented
#    actions_df[IPWS_COL] = ipws       #JN
#    
#   # if modified_probs_exist and config[read_config.SAVE_MODIFIED_ACTIONS_FILE_HEADER]:
#    if modified_probs_exist: #JN SHOULD NOT GET HERE TODO DELETE
#        prob_df = pd.DataFrame.from_records(saved_probs)
#        actions_df.loc[:,[H_ALGO_PROB_BEST_ACTION.format(1),H_ALGO_PROB_BEST_ACTION.format(2)]] = prob_df.values
#        actions_df[H_ALGO_NUM_TRIALS] = num_samples
#        actions_file_first_line = ""
#        with open(actions_infile) as infile:
#            actions_file_first_line = infile.readline()
#        with open(actions_infile,'w') as outfile:
#            outfile.write(actions_file_first_line)
#        print("saving to ", actions_infile)
#        actions_df.to_csv(actions_infile, mode='a', index=False)
        
    #---------------
#
  #  actions_df[IPWS_COL] = ipws
    
    ipw_means = []
    means = []
    ipw_terms = []
    ht_means = []
    sample_sizes = []

    for action in range(num_actions):
        cur_action = actions_df.loc[actions_df.loc[:,H_ALGO_ACTION] == (action + 1)] #inner part is idx's where chosen_action col == action+1 (ie. 1 or 2)
        #Outer looks at all cols, but only rows where action was chosen
        sample_sizes.append(len(cur_action))
        ipw_row[NUM_SAMPLES_BY_ACTION_HEADER.format(action + 1)] = cur_action.shape[0]
        if sum(cur_action.loc[:,IPWS_COL]) == 0:
            print("problem")
        if cur_action.shape[0] == 0:
            ipw_mean = float("nan")
            ipw_row[IPW_MEAN_HEADER.format(action + 1)] = ipw_mean
            ipw_means.append(ipw_mean)
            
            ht_mean = float("nan")
            ipw_row[HT_MEAN_HEADER.format(action + 1)] = ht_mean
            ht_means.append(ht_mean)
            
            ipw_terms.append(sum(cur_action.loc[:,IPWS_COL] * cur_action.loc[:,H_ALGO_OBSERVED_REWARD]))
            mean = float("nan")
            ipw_row[MLE_MEAN_HEADER.format(action + 1)] = mean
            means.append(mean)
        else:
            ipw_mean = sum(cur_action.loc[:,IPWS_COL] * cur_action.loc[:,H_ALGO_OBSERVED_REWARD]) / sum(cur_action.loc[:,IPWS_COL])
            ipw_row[IPW_MEAN_HEADER.format(action + 1)] = ipw_mean
            ipw_means.append(ipw_mean)
            
#            ht_mean = 1/(len(all_models[0]) - 1) * sum(cur_action.loc[:,IPWS_COL] * cur_action.loc[:,H_ALGO_OBSERVED_REWARD])
#            ipw_row[HT_MEAN_HEADER.format(action + 1)] = ht_mean
#            ht_means.append(ht_mean)
            
            ipw_terms.append(sum(cur_action.loc[:,IPWS_COL] * cur_action.loc[:,H_ALGO_OBSERVED_REWARD]))
            mean = np.mean(cur_action.loc[:,H_ALGO_OBSERVED_REWARD])
            ipw_row[MLE_MEAN_HEADER.format(action + 1)] = mean
            means.append(mean)
            
    sample_size_1 = sample_sizes[0]
    sample_size_2 = sample_sizes[1]
#    SE_ipw = np.sqrt(ipw_means[0]*(1 - ipw_means[0])/sample_size_1 + ipw_means[1]*(1 - ipw_means[1])/sample_size_2)
#    wald_type_stat_ipw = (ipw_means[0] - ipw_means[1])/SE_ipw #(P^hat_A - P^hat_b)/SE
#    #print('wald_type_stat:', wald_type_stat)
#    wald_pval_ipw = (1 - stats.norm.cdf(np.abs(wald_type_stat_ipw)))*2 #Two sided, symetric, so compare to 0.05
##
    
#    ipw_row[MAX_WEIGHT_COUNT_HEADER] = max_weights
#    print("means:", means)
#    print("ipw_means:", ipw_means)
#    print("ht_means:", ht_means)
#
#    print("diff in means:", means[0] - means[1])
#    print("diff in ipw_means:", ipw_means[0] - ipw_means[1])
#    print("ipw_terms:", 1/actions_df.shape[0] *(ipw_terms[0]-ipw_terms[1]))
#    print("count of max_weights:", max_weights, "/", len(cache_keys) - 1)
    
    mean_1_ipw, mean_2_ipw = np.array(ipw_means[0]), np.array(ipw_means[1])
    
    return mean_1_ipw, mean_2_ipw



def calculate_ipw_means_step_size(actions_root, num_samples, num_actions = 2, cached_probs={}, 
                  prior = [1,1], binary_rewards = True, \
                  config = {}, n = None,\
                  num_sims = None, batch_size = None, no_effect = True, effect_size = None):
    """
    Computes assignment probabilities, sets these to column 'ProbAction{}IsBest'
    Draws num_samples from a given model to determine assignment probabilties
    
    Some unused args from original code
    """
    
    assert num_actions == 2
    read_config.apply_defaults(config)
    match_num_samples = config[read_config.MATCH_NUM_SAMPLES_HEADER]
    smoothing_adder = config[read_config.SMOOTHING_ADDER_HEADER]
    max_weight = config[read_config.MAX_WEIGHT_HEADER]

    if no_effect:
        step_sizes = [int(np.ceil(n/2)), int(n), int(2*n), int(4*n)]
    else:
        nobs_total = smp.GofChisquarePower().solve_power(effect_size = effect_size, nobs = None, n_bins=(2-1)*(2-1) + 1, alpha = DESIRED_ALPHA, power = DESIRED_POWER)
#         print("Calculated nobs for effect size:", nobs_total)
        n = math.ceil(nobs_total)
        step_sizes = [math.ceil(n/2), n, 2*n, 4*n]
    step_sizes = [n]
    fig, ax = plt.subplots(1,4)
    ax = ax.ravel()
    
    i=0
  #  ipw_t1_list = [] #type 1 error rate per num partic. note is power when there is effect, is just proprtion reject null
    ipw_mean_1_list = []
    ipw_mean_2_list = []
    for num_steps in step_sizes:
        wald_pval_ipw_simlist = []
        wald_type_stat_ipw_simlist = []
        mean_1_per_sim_ipw = []
        mean_2_per_sim_ipw = []
        for sim_count in range(num_sims):
           # print(sim_count)
            
            actions_infile = actions_root + "/tbb_actions_{}_{}.csv".format(num_steps, sim_count)
            actions_df = pd.read_csv(actions_infile,skiprows=1)
            max_weights = 0
           # print(actions_df)
            if binary_rewards:
                all_models, cache_keys = create_models_binary(actions_df, prior, num_actions)
                
        
            else:
                all_models, cache_keys = create_models_normal(actions_df, prior, num_actions)
                
            
            mean_1_ipw, mean_2_ipw = calculate_ipw_means(actions_infile, num_samples, num_actions = 2, cached_probs={}, 
              prior = [1,1], binary_rewards = True, config = {})
            
            mean_1_per_sim_ipw.append(mean_1_ipw)
            mean_2_per_sim_ipw.append(mean_2_ipw)
        
        ipw_mean_1_across_sims = np.array(mean_1_per_sim_ipw)
        ipw_mean_2_across_sims = np.array(mean_2_per_sim_ipw)
        
#            wald_pval_ipw_simlist.append(wald_pval_ipw)
#            wald_type_stat_ipw_simlist.append(wald_type_stat_ipw)
#        wald_pval_ipw_simlist = np.array(wald_pval_ipw_simlist)
#        wald_type_stat_ipw_simlist = np.array(wald_type_stat_ipw_simlist)
#
#        num_rejected = np.sum(wald_pval_ipw_simlist < .05)
#        ipw_t1 = num_rejected/num_sims
#        
#        num_rejected_ipw_ppf = np.sum(np.abs(wald_type_stat_ipw_simlist) > stats.norm.ppf(0.975)) #sanity check
#        print("working on ", actions_infile)
#        print("num rejected same with and without pval:" ,num_rejected_ipw_ppf == num_rejected)
#        print(ipw_t1)
#       # probs_per_sim
#        ipw_t1_list.append(ipw_t1)
    
#        ipw_mean_1_list = np.array(ipw_mean_1_list)
#        ipw_mean_2_list = np.array(ipw_mean_2_list)
        
    
        print("saving to:", actions_root + "-ipw_mean_1_{}_participants.npy".format(num_steps))
        np.save(actions_root + "-ipw_mean_1_{}_participants.npy".format(num_steps), ipw_mean_1_across_sims)
        
        print("saving to:", actions_root + "-ipw_mean_2_{}_participants.npy".format(num_steps))
        np.save(actions_root + "-ipw_mean_2_{}_participants.npy".format(num_steps), ipw_mean_2_across_sims)
    #print("g")


def parse_dir(root, root_cutoffs, num_sims, outcomes_title_dict, sample_size_dict, experiment_title, condition_header):
    #num_sims = 1000
    #sims_dir = root + "/num_sims={}".format(num_sims)
    sims_dir = root.format(num_sims)
   

    print(os.path.isdir(sims_dir))
    sims_dir_cutoffs = root_cutoffs.format(num_sims)
    #n = 302
    #../../Outfile/ParamsBased/numSims5000motivate_final//Y1/bbEqualMeansEqualPriorburn_in_size-batch_size=17-17BB0N348Df.pkl#
    #n_list = [302]

    outcomes_list = ["Y1", "Y2", "Y3", "Y4"] #348 640 460
    prob_success_dict = {"Y1":[0.5, 0.75], "Y2":[0.5, 0.75], "Y3":[0.5, 0.75], "Y4":[0.25,0.5,0.75]}

    i = 0
    bs_prop_list =[0.05, 0.10, 0.25]
    
    
    for outcome in outcomes_list: #one plot per outcome, and also one table
            n=sample_size_dict[outcome]
            fig, ax = plt.subplots(3,1)
            #fig, ax = plt.subplots()
            fig.set_size_inches(17.5, 13.5)
            ax = ax.ravel()

            outcome_dir = sims_dir + "/{}/".format(outcome)
            outcome_dir_cutoffs = sims_dir_cutoffs + "/{}/".format(outcome)
            i = 0#for subplots
        
            for center in ["0pt25","0pt5","0pt75"]:
                
           
                outcome_dir_cutoffs_center = outcome_dir_cutoffs + "/"+ "center{}".format(center) + "/"
                print(os.path.isdir(outcome_dir))
            
                for bs_prop in bs_prop_list:

                    #bs = int(np.floor(bs_prop*n))
                    n=sample_size_dict[outcome]
                    bs = int(np.floor(n*bs_prop))
                    print("BS = ", bs, n*bs_prop)
                    print("---------------#-------")

                    actions_root = outcome_dir_cutoffs_center + "/bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}".format(bs,bs)
                    #outcome_dir + "/bbEqualMeansUniformburn_in_size-batch_size={}-{}".format(bs,bs)

                    config = {}
                    read_config.apply_defaults(config)
    
                    is_binary = True
                    prior = [1,1]
        
                    cached_probs = {}

                    calculate_ipw_means_step_size(actions_root = actions_root, num_samples=1000, num_actions = 2, cached_probs = {}, \
                          prior = prior, binary_rewards = is_binary, config = config, n = n, num_sims = num_sims, batch_size = bs)



#REMEMBER TO CHANGE "/" to ":"
num_sims=5000
root_dummy = "../2019-12-06_16:57:17NoEffect"

sample_size_dict = {"Y1":348,"Y2":658,"Y3":478, "Y4":657}
outcomes_title_dict = {"Y1": "Student success on the 2nd attempt at problem P1 for week 10", \
"Y2": "Student success on the 1st attempt at problem P2 for week 10", \
"Y3":"Student success on 1st attempt at problem P1 for week 12",\
"Y4":"Whether student succeeded in a maximum of 2 attempts at problem P3 on week 10"}


#Effect-----------------
#Extra Problems-----------
root = '../Outfile/ParamsBased/EffectnumSims{}problem_print_outputRedo/'
root_cutoffs = '../Outfile/ParamsBased/numSims{}ForCutoffsCenters/'

experiment_title = "Extra Problem"
condition_header = "problem_print_output"
parse_dir(root, root_cutoffs, num_sims, outcomes_title_dict, sample_size_dict, experiment_title, condition_header)


