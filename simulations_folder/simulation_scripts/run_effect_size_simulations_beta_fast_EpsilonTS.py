'''
Created on Jul 23, 2020

@author: arghavanmodiri
'''
import argparse 
import math
import random
import sys
import time
import logging

import scipy.stats
sys.path.insert(1, '../../louie_experiments/')
import beta_bernoulli
import effect_size_sim_output_viz
from forced_actions import forced_actions
import generate_single_bandit
import get_assistments_rewards
import numpy as np
import pandas as pd
import reorder_samples_in_rewards
import run_effect_size_simulations
import statsmodels.stats.api as sms
import statsmodels.stats.power as smp
import thompson_policy
import output_format


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

PRIOR_PROPORTION_DIFFERENCE = .5 # When making arms that don't have the prior in the center what proportion of the remaining space to use for the arms

action_header ='AlgorithmAction'
obs_reward_header = 'ObservedRewardofAction'
ppd_exp_header = "IsExploring"


def make_stats_row_from_df(simulations_df, include_power, action_count, effect_size = None, alpha = None):
    '''Calculates output statistics given the data frame simulations_df. If include_power, includes the power calculation.
    efffect_size and alpha are only required/used if power is calculated
    '''
    #REPLACE cur_data with simulations_df
    #number of actions! action_count
    step_size = max(simulations_df.index) + 1 #will be 4n for instance
    n = step_size/4 #assumes set to 4n
    n_size_list = [math.ceil(n/2), int(n), int(2*n), int(4*n)] 
    print("step_size", step_size)
    print("n_size_list", n_size_list)
    trials_count = len(simulations_df)
    print("trials_count", trials_count)#will go to end, so 4n*num_sims
    print(simulations_df.columns)

    all_rows = []


    for n_size in n_size_list:
        sim_num = 0
        for idx in range(0, trials_count, step_size):
            one_sim_df = simulations_df[idx:idx + n_size].copy()
            assert(len(one_sim_df) == n_size)

            cur_row = {}

            prop_exploring_ppd_cuml = np.sum(one_sim_df[ppd_exp_header])/n_size #cumulative
#            print(one_sim_df["SampleNumber"])
#            print("n_size", n_size, exploring_this_n)
            exploring_this_n = one_sim_df[one_sim_df["SampleNumber"] == n_size - 1][ppd_exp_header].iloc[0]#snap shot, conver to idx

            sample_sizes = np.array([])
            successes = np.array([])
            means = np.array([])
            np.array([])
            np.array([])
            for i in range(1, action_count+1):
                sample_sizes = np.append(sample_sizes, np.sum(one_sim_df[action_header] == i))
                successes = np.append(successes, np.sum(one_sim_df[one_sim_df[action_header] == i][obs_reward_header]))
                means = np.append(means, np.mean(one_sim_df[one_sim_df[action_header] == i][obs_reward_header]))

            #calculate sample size and mean
            for i in range(action_count):
                cur_row['sample_size_{}'.format(i+1)] = sample_sizes[i]
                cur_row['mean_{}'.format(i+1)] = means[i]

            if action_count == 2:
                #SE = sqrt[(P^hat_A*(1-P^hat_A)/N_A + (P^hat_B*(1-P^hat_B)/N_B]
                SE = np.sqrt(means[0]*(1 - means[0])/sample_sizes[0] + means[1]*(1 - means[1])/sample_sizes[1])
                wald_type_stat = (means[0] - means[1])/SE #(P^hat_A - P^hat_b)/SE

                #print('wald_type_stat:', wald_type_stat)
                #Two sided, symetric, so compare to 0.05
                wald_pval = (1 - scipy.stats.norm.cdf(np.abs(wald_type_stat)))*2 

                cur_row['wald_type_stat'] = wald_type_stat
                cur_row['wald_pval'] = wald_pval

            #calculate total reward
            cur_row['total_reward'] = np.sum(one_sim_df[obs_reward_header])
            #calculate power
            cur_row['ratio'] = sample_sizes[0] / sample_sizes[1]
            if include_power:
                cur_row['power'] = smp.GofChisquarePower().solve_power(effect_size,
                    nobs = sum(sample_sizes), n_bins=(2-1)*(2-1) + 1, alpha = alpha)
            cur_row['actual_es'] = calculate_effect_size(sample_sizes, successes)

            #calculate chi squared contingency test
            table = sms.Table(np.stack((successes,sample_sizes - successes)).T)
            rslt = table.test_nominal_association()
            cur_row['stat'] = rslt.statistic
            cur_row['pvalue'] = rslt.pvalue
            cur_row['df'] = rslt.df
            # Added to match normal rewards
            cur_row['statUnequalVar'] = cur_row['stat']
            cur_row['pvalueUnequalVar'] = cur_row['pvalue']
            cur_row['dfUnequalVar'] = cur_row['df']
            cur_row['num_steps'] = max(one_sim_df.index) + 1
    #        cur_row['num_steps'] = n_size 
            cur_row['sim'] = sim_num

            cur_row["prop_exploring_ppd_cuml"] = prop_exploring_ppd_cuml
            cur_row["exploring_ppd_at_this_n"] = exploring_this_n

            all_rows.append(cur_row)
            sim_num += 1

    return all_rows



def calculate_statistics_from_sims(simulations_dfs_list, effect_size,
    action_count, alpha = 0.05):
    '''
    - Output that we'll look at:
    -- Given the actual average ratios of conditions, what is our average power
        (calculated using formulas above)?
    -- Actually conducting a t-test on our observations, what are our average
        statistics and p-values? Calculate the rate of rejecting the null
        hypothesis for the simulations (we'd expect it to be .8, since that's
        what power means).
    -- How large is our actual effect size on average, given that we have unequal samples?
    '''
    columns=['num_steps', 'sim', 'sample_size_1', 'sample_size_2', 'mean_1',
        'mean_2', 'total_reward', 'ratio', 'power', 'actual_es', 'stat',
        'pvalue', 'df','statUnequalVar','pvalueUnequalVar', 'dfUnequalVar',
        'wald_pval', 'wald_type_stat', "prop_exploring_ppd_cuml", "exploring_ppd_at_this_n"]

    all_rows_all_n = []
    for simulations_df in simulations_dfs_list:
        stats_rows_list = make_stats_row_from_df(simulations_df, True,
            action_count, effect_size, alpha)
        all_rows_all_n.extend(stats_rows_list)

    all_rows_all_n_df = pd.DataFrame(all_rows_all_n, columns=columns)

    return all_rows_all_n_df


def calculate_effect_size(ns, successes):
    '''
    Calculates the actual effect size given that we have ns[i] observations for each condition i,
    and successes[i] successes were observed in condition i.
    '''
    ns = np.array(ns)
    successes = np.array(successes)
    succ_prop = successes / sum(ns)
    fail_prop = (ns - successes) / sum(ns)
    prob_table_h1 = np.stack((succ_prop, fail_prop)).T # This is equivalent to P in ES.w2 in R
    pi = np.sum(prob_table_h1,axis=0)
    pj = np.sum(prob_table_h1,axis=1)
    prob_table_h0 = np.outer(pi, pj).T
    w = math.sqrt(sum(sum((prob_table_h1-prob_table_h0)**2/prob_table_h0)))
    return w


def get_output_filename(outfile_directory, num_steps, sim_num=None, mode=''):
    '''
    Returns the name of the file that will have the actions taken for sim_num that has num_steps steps.
    '''
    if mode=='uniform':
        t = 'u'
    else:
        t = 'b'

    separator = '/'
    if outfile_directory.endswith('/'):
        separator = ''
    results_data_file = outfile_directory + separator + 'tb{0}_actions_{1}'
    return results_data_file.format(t, num_steps)


def run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory,
    successPrior = 1, failurePrior = 1, softmax_beta = None,
    reordering_fn = None, forceActions = 0, batch_size = 1, burn_in_size = 1,
    random_dur=0, random_start=0, mode='', epsilon = 0.1, resample = True):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy.
    '''
    csv_output_file_names = []
    sim_results_dfs_list = []

    for num_steps in step_sizes:
        sim_results = []
        for i in range(num_sims):
            if forceActions != 0:
                forced = run_effect_size_simulations.make_forced_actions(len(prob_per_arm), num_steps, forceActions)
            else:
                forced = forced_actions()

            if softmax_beta != None:
                # reorder rewards
                raise ValueError("softmax_beta is not supported in fast mode.")

            if mode=='uniform':
                models = [beta_bernoulli.BetaBern(success=1, failure=1) for _ in range(len(prob_per_arm))]
                random_dur = num_steps
            else:
                models = [beta_bernoulli.BetaBern(success=successPrior, failure=failurePrior) for _ in range(len(prob_per_arm))]


            sim_result, column_names,_ = \
                thompson_policy.two_phase_random_thompson_policy(
                            prob_per_arm=prob_per_arm,
                            users_count=num_steps,
                            random_dur=random_dur,#100,
                            models=models,
                            random_start=random_start,
                            action_mode=thompson_policy.ActionSelectionMode.prob_is_best,
                            relearn=True,
                            forced = forced,
                            batch_size = batch_size, epsilon=epsilon)

            sim_results.extend(sim_result)

        sim_results_df = pd.DataFrame(sim_results, columns=column_names)
        sim_results_df.index = [idx for idx in range(num_steps)]*num_sims
        sim_results_dfs_list.append(sim_results_df)

        cur_output_file = get_output_filename(outfile_directory, num_steps, None, mode)
        csv_output_file_names.append(cur_output_file)

    return sim_results_dfs_list, csv_output_file_names


def get_prob_per_arm_from_effect_size(effect_size, center = 0.5):
    '''
    Calculates the probability of success on each arm that would be needed to get the given effect_size.
    Assumes the sample sizes will be equal, and that we have two arms, one with probability .5 + x, the other
    with probability .5 - x, for some x.
    '''
    x = effect_size / 2
    return [center + x, center - x]


def main():
    start_time = time.time()
    outfile_directory = sys.argv[3]

    random_dur_m = 0
    random_start_r = 0
    recalculate_bandits = True
    num_arms = 2

    #batch_size = 1.0
 #   if len(sys.argv) > 5:
  #      epsilon = float(sys.argv[5])
   #     print("epsilon", epsilon)
    if "epsilon=" in outfile_directory:
        print(outfile_directory)
        epsilon = float(outfile_directory.split("epsilon=")[-1].split("/")[0].strip("="))
        #c = float(outfile_directory.split("=c=")[-1])
        print("epsilon", epsilon)
    
    num_sims = int(sys.argv[2])
    burn_in_size, batch_size = int(outfile_directory.split("=")[-1].split('-')[0]), int(outfile_directory.split("=")[-1].split('-')[1])
    print("burn_in_size, batch_size", burn_in_size, batch_size)

    # if sys.argv[1] has a comma, just use the result as probability per arm
    if "," in sys.argv[1]:
        if sys.argv[1].count(",") ==1:
            # specifying probability per arm but not effect size
            prob_per_arm = [float(armProb) for armProb in sys.argv[1].split(",")]
            effect_size = 0 # Note: This will be wrong if arm probs aren't equal!
        else:
            # specifying probability per arm as first two arguments, and then effect size
            numeric_arguments = [float(armProb) for armProb in sys.argv[1].split(",")];
            prob_per_arm = numeric_arguments[:2] # first two are arm probabilities
            effect_size = numeric_arguments[2] # final is effect size
        # We also need to specify n in this case for deciding on step sizes
        n = int(sys.argv[6])
    else:
        # We just need effect size for this calculation
        effect_size = float(sys.argv[1].split("-")[0])
        center = float(sys.argv[1].split("-")[1])
        prob_per_arm = get_prob_per_arm_from_effect_size(effect_size, center)
        # Assumes we have two arms
        nobs_total = smp.GofChisquarePower().solve_power(effect_size, n_bins=(2-1)*(2-1) + 1, alpha = DESIRED_ALPHA, power = DESIRED_POWER)
        #print("Calculated nobs for effect size:", nobs_total)
        n = math.ceil(nobs_total)
        print("center", center)

    '''
    These differ from the version for normal because in normal,
    n represented size for one cond rather than overall size
    step_sizes = [math.ceil(n/2), n, 2*n, 4*n] 
    '''

    #Arghavan: Just run simulation for n
    step_sizes = [4*n]

    print("prob_per_arm", prob_per_arm)
    if len(sys.argv) > 7 and sys.argv[7].startswith("forceActions"):
        run_effect_size_simulations.FORCE_ACTIONS = True
        num_to_force = float(sys.argv[7].split(",")[1])
    else:
        num_to_force = 1#force one action from each arm to avoid nan means 
    
    bandit_type = "Thompson"
    bandit_type_prefix = 'BB'
    if len(sys.argv) > 4:
        bandit_type = sys.argv[4]
    if bandit_type == "uniform":
        bandit_type_prefix = "BU"# Bernoulli rewards, uniform policy

    reorder_rewards = False
    softmax_beta = None
    reordering_fn = None
    if len(sys.argv) > 7 and not sys.argv[7].startswith("forceActions"):
        # softmax beta for how to reorder rewards
        reorder_rewards = True
        softmax_beta = float(sys.argv[7])
        reordering_fn = reorder_samples_in_rewards.order_by_named_column('Action1OracleActualReward')
        if len(sys.argv) > 8:
            reordering_fn_specifier = sys.argv[8]
            reordering_fn = reorder_samples_in_rewards.get_reordering_fn(reordering_fn_specifier)

    prior_params = None
    if recalculate_bandits:
        if bandit_type == "uniform":
            results_dfs_list, results_output_names = run_simulations(
                    num_sims, prob_per_arm, step_sizes, outfile_directory,
                    forceActions = num_to_force, mode='uniform')
        else:
            if len(sys.argv) > 5:
                if sys.argv[5] == "armsHigh":
                    # Arms should be higher than the prior
                    priorProportionOnSuccess = min(prob_per_arm)*PRIOR_PROPORTION_DIFFERENCE
                elif sys.argv[5] == "armsLow":
                    # Arms should be lower than the prior
                    priorProportionOnSuccess = 1 - (1-max(prob_per_arm))*PRIOR_PROPORTION_DIFFERENCE
                else:
                    # Prior should be uniform (in between arms)
                    priorProportionOnSuccess = .5
                # Make sure the prior sums to 2, mirroring the successes/failures of uniform prior
                prior_params = [priorProportionOnSuccess*2, 2-priorProportionOnSuccess*2]
                print("Prior params: ", prior_params)

                results_dfs_list, results_output_names = run_simulations(
                    num_sims, prob_per_arm, step_sizes,
                    outfile_directory, prior_params[0], prior_params[1],
                    softmax_beta = softmax_beta, reordering_fn = reordering_fn,
                    forceActions = num_to_force, batch_size = batch_size,
                    burn_in_size = burn_in_size, random_dur=random_dur_m,
                    random_start=random_start_r, epsilon=epsilon)
            else:
                results_dfs_list, results_output_names = run_simulations(num_sims, prob_per_arm, step_sizes,
                    outfile_directory, forceActions = num_to_force,
                    batch_size = batch_size, burn_in_size = burn_in_size, epsilon=epsilon)


    outfile_prefix = outfile_directory  + bandit_type_prefix + str(effect_size)
    if effect_size == 0:
        # Then include the n  in the prefix
        outfile_prefix += "N" + str(n) 

    for results_df, results_output_name in zip(results_dfs_list, results_output_names):
        results_df['SampleNumber'] = results_df.index

        if num_sims <= 2:
            results_df.to_csv('{}_sims={}_m={}.csv'.format(results_output_name, num_sims, random_dur_m), index=False)
#Not saving for now
#        results_df.to_csv('{}_sims={}_m={}.csv.gz'.format(results_output_name, num_sims, random_dur_m), compression = "gzip", index=False)

    stats_df = calculate_statistics_from_sims(results_dfs_list,effect_size, num_arms, alpha = 0.05)
    stats_df.to_pickle(outfile_prefix + 'Df_sim={}_m={}_r={}.pkl'.format(num_sims, random_dur_m,random_start_r))

    end_time = time.time()
    print('Execution time = %.6f seconds' % (end_time-start_time))




if __name__=="__main__":
    main()
