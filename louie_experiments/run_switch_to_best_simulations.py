'''
Created on May 22, 2018

Runs simulations either for a fixed sample size
with uniform sampling and then for another fixed n samples
either goes to the best arm or continues uniform random if
there is no statistical difference. If bandit sampling, just does the normal
thing.

(Intended to write data in a way that the same scripts for analyzing other
simulations can be used.)

@author: rafferty
'''
import math
import random
import sys
import os.path

from scipy import stats
import statsmodels.stats.power

import effect_size_sim_output_viz
from forced_actions import forced_actions
import generate_single_bandit
import get_assistments_rewards
import ng_normal
import numpy as np
import pandas as pd
import reorder_samples_in_rewards
import statsmodels.stats.api as sms
import statsmodels.stats.power as smp
import run_effect_size_simulations
import thompson_ng_policy
import run_effect_size_simulations_beta
import beta_bernoulli
import thompson_policy

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05
FORCE_ACTIONS = False;
action_header ='AlgorithmAction'
obs_reward_header = 'ObservedRewardofAction'

def make_forced_actions(num_actions, num_steps, num_actions_to_force = 5):
    '''
    Returns a forced actions object that forces an equal number of
    each action and where the number of forced actions may be based
    on the total number of steps. If num_actions_to_force is < 1,
    treats it as a proportion of the total number of steps. If a proportion
    is used, rounds up to next full trial. (E.g., the fewest number of forced actions
    of each type you'll ever have with a proportion is 1.)
    '''
#     print("num_actions:",num_actions)
#     print("num_steps:", num_steps)
#     print("num_actions_to_force:", num_actions_to_force)

    if num_actions_to_force < 1:
        num_actions_to_force = int(math.ceil(num_steps*num_actions_to_force))
    else:
        num_actions_to_force = int(math.ceil(num_actions_to_force))
    forced_action_counts = [num_actions_to_force for _ in range(num_actions)]
    action_list = [i for i in range(len(forced_action_counts)) for _ in range(forced_action_counts[i])]
    forced = forced_actions(actions=action_list)
    return forced

def calculate_by_trial_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, alpha = 0.05):
    '''
    Similar to the non-by_trial version, but adds in a column for trial number and repeatedly analyzes the first
    n steps of each trial, for n = 1:N.
    '''
    rows =[]
    for num_steps in step_sizes: 
        for i in range(num_sims):
            output_file = pd.read_csv(get_output_filename(outfile_directory, num_steps, i),skiprows=1)
            for step in range(effect_size_sim_output_viz.TRIAL_START_NUM, num_steps + 1):
                cur_data = output_file[:step]
                cur_row = make_stats_row_from_df(cur_data, False)
                cur_row['num_steps'] = num_steps
                cur_row['sim'] = i
                cur_row['trial'] = step          
                rows.append(cur_row)
    # Now do some aggregating
    df = pd.DataFrame(rows, columns=['num_steps', 'sim', 'trial', 'sample_size_1', 'sample_size_2', 'mean_1','mean_2', 'total_reward', 'ratio', 'actual_es', 'stat', 'pvalue', 'df'])
    
    return df

def create_arm_stats_by_step(outfile_directory, num_sims, num_steps, num_arms = 2):
    '''
    Creates a data frame that has the average arm estimates for each arm at each point in the simulation.
    Includes the estimated mean and variance for each arm, as well as what simulation and trial we're on.
    '''
    mu_template = 'Action{0}EstimatedMu'
    var_template =  'Action{0}EstimatedArmVariance'
    mus = [mu_template.format(i) for i in range(1, num_arms + 1)]
    variances = [var_template.format(i) for i in range(1, num_arms + 1)]

    arm_mean_columns = ['mean_arm_' + str(i) for i in range(1, num_arms + 1)]
    arm_var_columns = ['var_arm_' + str(i) for i in range(1, num_arms + 1)]
    
    all_dfs = []
    for i in range(num_sims):
        output_file = pd.read_csv(get_output_filename(outfile_directory, num_steps, i),skiprows=1)
        mean_dfs = [output_file[mu_header] for mu_header in mus]
        all_means_df = pd.DataFrame({arm_mean_column: col for arm_mean_column, col in zip(arm_mean_columns,mean_dfs)})
        var_dfs = [output_file[var_header] for var_header in variances]
        all_vars_df = pd.DataFrame({arm_var_column: col for arm_var_column, col in zip(arm_var_columns,var_dfs)})
        means_and_var_df = pd.concat([all_means_df,all_vars_df], axis=1, join="inner")
        means_and_var_df.insert(0,'num_steps',num_steps)
        means_and_var_df.insert(0,'sim',i)
        means_and_var_df.insert(0,'trial',range(0, all_vars_df.shape[0]))
        all_dfs.append(means_and_var_df)

    # Now do some aggregating


    df = all_dfs[0].append(all_dfs[1:])
    return df


def make_stats_row_from_df(cur_data, include_power, effect_size = None, alpha = None):
    '''Calculates output statistics given the data frame cur_data. If include_power, includes the power calculation.
    efffect_size and alpha are only required/used if power is calculated
    '''
    cur_row = {}
    sample_sizes = [np.sum(cur_data[action_header] == i) for i in range(1,3)]
    #calculate sample size and mean
    cur_row['sample_size_1'] = sample_sizes[0]
    cur_row['sample_size_2'] = sample_sizes[1]
    cur_row['mean_1'] = np.mean(cur_data[cur_data[action_header] == 1][obs_reward_header])
    cur_row['mean_2'] = np.mean(cur_data[cur_data[action_header] == 2][obs_reward_header])
    
    #calculate total reward
    cur_row['total_reward'] = np.sum(cur_data[obs_reward_header])
    
    #calculate power
    cur_row['ratio'] = sample_sizes[0] / sample_sizes[1]
    if include_power:
        cur_row['power'] = statsmodels.stats.power.tt_ind_solve_power(effect_size, cur_row['sample_size_1'], alpha, None, cur_row['ratio'])
    cur_row['actual_es'] = calculate_effect_size(cur_data[cur_data[action_header] == 1][obs_reward_header], cur_data[cur_data[action_header] == 2][obs_reward_header])
    
    #calculate ttest
    comparer = sms.CompareMeans(sms.DescrStatsW(cur_data[cur_data[action_header] == 1][obs_reward_header]),
                                sms.DescrStatsW(cur_data[cur_data[action_header] == 2][obs_reward_header]))
    cur_row['stat'], cur_row['pvalue'], cur_row['df'] = comparer.ttest_ind(usevar = 'pooled')
    cur_row['statUnequalVar'], cur_row['pvalueUnequalVar'], cur_row['dfUnequalVar'] = comparer.ttest_ind(usevar = 'unequal')

#     cur_row['statSP'], cur_row['pvalueSP'] = stats.ttest_ind(cur_data[cur_data[action_header] == 1][obs_reward_header], cur_data[cur_data[action_header] == 2][obs_reward_header], equal_var = ASSUME_EQUAL_VAR)
#     cur_row['statOppSP'], cur_row['pvalueOppSP'] = stats.ttest_ind(cur_data[cur_data[action_header] == 1][obs_reward_header], cur_data[cur_data[action_header] == 2][obs_reward_header], equal_var = not ASSUME_EQUAL_VAR)
    
    return cur_row

def calculate_effect_size(rewards_arm_1, rewards_arm_2):
    '''
    Calculates Cohen's d given the rewards observed for each arm.
    '''
    sample_std_1 = np.std(rewards_arm_1, ddof=1)
    sample_std_2 = np.std(rewards_arm_2, ddof=1)
    pooled_std = math.sqrt((len(rewards_arm_1)*sample_std_1**2 + len(rewards_arm_2)*sample_std_2**2)/(len(rewards_arm_1) + len(rewards_arm_2) - 2))
    cohens_d =(np.mean(rewards_arm_1) - np.mean(rewards_arm_2)) / pooled_std
    return cohens_d

def calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, 
                                   switch_to_best_if_nonsig, switch_points, 
                                   alpha = 0.05, is_binary = False):
    '''
    - Output that we'll look at:
    -- Given the actual average ratios of conditions, what is our average power (calculated using formulas above)?
    -- Actually conducting a t-test on our observations, what are our average statistics and p-values? Calculate the rate of rejecting the null hypothesis for the simulations (we'd expect it to be .8, since that's what power means).
    -- How large is our actual effect size on average, given that we have unequal samples?
    '''
    # Go through each file to collect the vectors of rewards we observed from each arm [i.e. condition]
    rows =[]
    for num_steps, num_steps_before_switch in zip(step_sizes,switch_points): 
        for i in range(num_sims):
            output_file_orig = pd.read_csv( get_output_filename(outfile_directory, num_steps, i),skiprows=1)
            stats_are_before_switch = [False, True]
            for before_switch_only in stats_are_before_switch:
                if before_switch_only:
                    output_file = output_file_orig.iloc[0:num_steps_before_switch,:]
                else:
                    output_file = output_file_orig
                if is_binary:
                    cur_row = run_effect_size_simulations_beta.make_stats_row_from_df(output_file, True, effect_size, alpha)
                else:
                    cur_row = make_stats_row_from_df(output_file, True, effect_size, alpha)
                cur_row['num_steps'] = num_steps
                cur_row['sim'] = i
                cur_row['switch_to_best_if_nonsig'] = switch_to_best_if_nonsig    
                cur_row['switch_point'] = num_steps_before_switch  
                cur_row['before_switch_only'] = before_switch_only                                            
                rows.append(cur_row)
    # Now do some aggregating
    df = pd.DataFrame(rows, columns=['num_steps', 'sim', 'sample_size_1', 'sample_size_2', 'mean_1','mean_2', 'total_reward', 'ratio', 'power', 'actual_es', 'stat', 'pvalue', 'df', 'statUnequalVar','pvalueUnequalVar', 'dfUnequalVar', 'switch_to_best_if_nonsig', 'switch_point'])#, 'before_switch_only'])

    return df

def get_rewards_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the rewards for sim_num that has num_steps steps.
    '''
    reward_data_file = outfile_directory + '/tng_rewards_{0}_{1}.csv'
    formatted_reward_data_file = reward_data_file.format(num_steps, sim_num)
    if not os.path.isfile(formatted_reward_data_file):
        reward_data_file = outfile_directory + '/tbb_rewards_{0}_{1}.csv'
        formatted_reward_data_file = reward_data_file.format(num_steps, sim_num)
    return formatted_reward_data_file

def get_reordered_rewards_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the rewards for sim_num that has num_steps steps.
    '''
    reward_data_file = outfile_directory + '/tng_rewards_reordered_{0}_{1}.csv'
    formatted_reward_data_file = reward_data_file.format(num_steps, sim_num)
    if not os.path.isfile(formatted_reward_data_file):
        reward_data_file = outfile_directory + '/tbb_rewards_reordered_{0}_{1}.csv'
        formatted_reward_data_file = reward_data_file.format(num_steps, sim_num)
    return formatted_reward_data_file

def get_output_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the actions taken for sim_num that has num_steps steps.
    '''
    results_data_file = outfile_directory + '/tng_actions_{0}_{1}.csv'
    formatted_results_data_file = results_data_file.format(num_steps, sim_num)
    if not os.path.isfile(formatted_results_data_file):
        results_data_file = outfile_directory + '/tbb_actions_{0}_{1}.csv'
        formatted_results_data_file = results_data_file.format(num_steps, sim_num)
    return formatted_results_data_file

def run_simulations(num_sims, mean_list, variance, step_sizes, outfile_directory, softmax_beta = None, reordering_fn = None, prior_mean = 0,  forceActions = 0):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy.
    '''

    for i in range(num_sims):
        for num_steps in step_sizes:
            if forceActions != 0:
                print("Forcing actions:", forceActions)
                forced = make_forced_actions(len(mean_list), num_steps, forceActions)
            else:
                forced = forced_actions()
            cur_reward_file = get_rewards_filename(outfile_directory, num_steps, i)
            # Check if they've passed in one variance for everything or multiple variances
            if not hasattr(variance, '__len__'):
                # only one variance - turn into a list
                variances = [variance]*len(mean_list)
            else:
                # multiple variances - pass straight through
                variances = variance
                
                
            generate_single_bandit.generate_normal_distribution_file(mean_list, 
                                                                    variances,
                                                                    num_steps,        
                                                                    cur_reward_file)
            if softmax_beta != None:
                # reorder rewards
                reordered_reward_file = get_reordered_rewards_filename(outfile_directory, num_steps, i)
                reorder_samples_in_rewards.reorder_rewards_by_quartile(cur_reward_file, 
                                                                       reordered_reward_file, 
                                                                       reordering_fn, 
                                                                       softmax_beta)
            else:
                reordered_reward_file = cur_reward_file
            cur_output_file = get_output_filename(outfile_directory, num_steps, i)
            models = [ng_normal.NGNormal(mu=prior_mean, k=1, alpha=1, beta=1) for _ in range(len(mean_list))]
            thompson_ng_policy.calculate_thompson_single_bandit(reordered_reward_file, 
                                         num_actions=len(mean_list), 
                                         dest= cur_output_file, 
                                         models=models, 
                                         action_mode=thompson_ng_policy.ActionSelectionMode.prob_is_best, 
                                         relearn=True,
                                         forced = forced)
            
def run_simulations_empirical_rewards(num_sims, reward_file, experiment_id, reward_header, is_cost, 
                                      outfile_directory, prior_mean = 0,  forceActions = 0, shuffle_data = False):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy. Assumes reward_file is formatted like ASSISTments data,
    where the reward is present under the column reward_header. Runs for as many steps as it's able
    to gain samples
    '''
    num_actions = 2
    max_steps = -1
    means = []
    variance = []
    for i in range(num_sims):
        arm_1_rewards, arm_2_rewards = get_assistments_rewards.read_assistments_rewards(reward_file, 
                                                                                        reward_header, 
                                                                                        experiment_id,
                                                                                        is_cost);
        if shuffle_data:
            random.shuffle(arm_1_rewards)
            random.shuffle(arm_2_rewards)
        max_steps = len(arm_1_rewards) + len(arm_2_rewards)
        means = [np.mean(arm_1_rewards), np.mean(arm_2_rewards)]
        variance= [np.var(arm_1_rewards), np.var(arm_2_rewards)]
        if forceActions != 0:
            print("Forcing actions:", forceActions)
            forced = make_forced_actions(num_actions, len(arm_1_rewards) + len(arm_2_rewards), forceActions)
        else:
            forced = forced_actions()

        
        cur_output_file = get_output_filename(outfile_directory, len(arm_1_rewards) + len(arm_2_rewards), i)
        models = [ng_normal.NGNormal(mu=prior_mean, k=1, alpha=1, beta=1) for _ in range(num_actions)]
        thompson_ng_policy.calculate_thompson_single_bandit_empirical_params(arm_1_rewards,
                                     arm_2_rewards,
                                     num_actions=num_actions, 
                                     dest= cur_output_file, 
                                     models=models, 
                                     action_mode=thompson_ng_policy.ActionSelectionMode.prob_is_best, 
                                     relearn=True,
                                     forced = forced)
    return max_steps, means, variance
            
def run_simulations_uniform_random(num_sims, mean_list, variance, steps_before_switch, steps_after_switch,
                                    outfile_directory, forceActions = 0, switch_to_best_if_nonsignificant = True):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Samples uniformly at random.
    '''

    for i in range(num_sims):
        if forceActions != 0:
            print("Forcing actions:", forceActions)
            forced = make_forced_actions(len(mean_list), steps_before_switch, forceActions)
        else:
            forced = forced_actions()
        cur_reward_file = get_rewards_filename(outfile_directory, steps_before_switch + steps_after_switch, i)
        # Check if they've passed in one variance for everything or multiple variances
        if not hasattr(variance, '__len__'):
            # only one variance - turn into a list
            variances = [variance]*len(mean_list)
        else:
            # multiple variances - pass straight through
            variances = variance
        generate_single_bandit.generate_normal_distribution_file(mean_list, 
                                                                variances,
                                                                steps_before_switch + steps_after_switch,        
                                                                cur_reward_file)
#        
        cur_output_file = get_output_filename(outfile_directory, steps_before_switch + steps_after_switch, i)
        models = [ng_normal.NGNormal(mu=0, k=1, alpha=1, beta=1) for _ in range(len(mean_list))]

        thompson_ng_policy.calculate_thompson_switch_to_fixed_policy(cur_reward_file, 
                                     num_actions=len(mean_list), 
                                     dest= cur_output_file, 
                                     num_actions_before_switch=steps_before_switch,
                                     models=models,
                                     switch_to_best_if_nonsignificant = switch_to_best_if_nonsignificant, 
                                     epsilon = 1.0,
                                     action_mode=thompson_ng_policy.ActionSelectionMode.prob_is_best, 
                                     forced = forced)
        
        
def run_simulations_uniform_random_binary(num_sims, prob_per_arm, steps_before_switch, steps_after_switch,
                                    outfile_directory, forceActions = 0, switch_to_best_if_nonsignificant = True):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Samples uniformly at random.
    '''
    num_steps = steps_before_switch + steps_after_switch

    for i in range(num_sims):
        if forceActions != 0:
            print("Forcing actions:", forceActions)
            forced = run_effect_size_simulations.make_forced_actions(len(prob_per_arm), num_steps, forceActions)
        else:
            forced = forced_actions()
        
        cur_reward_file = get_rewards_filename(outfile_directory, num_steps, i)
        generate_single_bandit.generate_file(np.array(prob_per_arm),
                                             num_steps,        
                                             cur_reward_file)
#        
        cur_output_file = get_output_filename(outfile_directory, num_steps, i)
        models = [beta_bernoulli.BetaBern(success=1, failure=1) for _ in range(len(prob_per_arm))]
        thompson_policy.calculate_thompson_switch_to_fixed_policy(cur_reward_file, 
                                     num_actions=len(prob_per_arm), 
                                     dest= cur_output_file, 
                                     num_actions_before_switch=steps_before_switch,
                                     models=models, 
                                     action_mode=thompson_policy.ActionSelectionMode.prob_is_best,
                                     epsilon = 1.0, 
                                     switch_to_best_if_nonsignificant=switch_to_best_if_nonsignificant,
                                     forced = forced)
            
#         cur_reward_file = get_rewards_filename(outfile_directory, steps_before_switch + steps_after_switch, i)
#         # Check if they've passed in one variance for everything or multiple variances
#         if not hasattr(variance, '__len__'):
#             # only one variance - turn into a list
#             variances = [variance]*len(mean_list)
#         else:
#             # multiple variances - pass straight through
#             variances = variance
#         generate_single_bandit.generate_normal_distribution_file(mean_list, 
#                                                                 variances,
#                                                                 steps_before_switch + steps_after_switch,        
#                                                                 cur_reward_file)
# #        
#         cur_output_file = get_output_filename(outfile_directory, steps_before_switch + steps_after_switch, i)
#         models = [ng_normal.NGNormal(mu=0, k=1, alpha=1, beta=1) for _ in range(len(mean_list))]
# 
#         thompson_ng_policy.calculate_thompson_switch_to_fixed_policy(cur_reward_file, 
#                                      num_actions=len(mean_list), 
#                                      dest= cur_output_file, 
#                                      num_actions_before_switch=steps_before_switch,
#                                      models=models,
#                                      switch_to_best_if_nonsignificant = switch_to_best_if_nonsignificant, 
#                                      epsilon = 1.0,
#                                      action_mode=thompson_ng_policy.ActionSelectionMode.prob_is_best, 
#                                      forced = forced)

    


def get_var_from_effect_size(mean1, mean2, effect_size):
    '''
    Calculates the variance that would be needed to get the given effect_size, given
    the means. Assumes the sample sizes will be equal.
    '''
    return abs(mean1-mean2)/effect_size

def get_mean_and_prior_from_string(string):
    if ',' in string: # setting the prior as well
        mean1, mu1 = [float(item) for item in string.split(',')]
    else:
        mean1 = float(string)
        mu1 = 0 # default to prior of 0
    return mean1, mu1

def mainBinaryRewards():
    print("Running binary")
    recalculate_bandits = True
    
    num_sims = int(sys.argv[2])
    outfile_directory = sys.argv[3]
    num_arms = 2
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
        effect_size = float(sys.argv[1])
        prob_per_arm = run_effect_size_simulations_beta.get_prob_per_arm_from_effect_size(effect_size)
        # Assumes we have two arms
        nobs_total = smp.GofChisquarePower().solve_power(effect_size, n_bins=(2-1)*(2-1) + 1, alpha = DESIRED_ALPHA, power = DESIRED_POWER)
#         print("Calculated nobs for effect size:", nobs_total)
        n = math.ceil(nobs_total)
#     step_sizes = [math.ceil(n/2), n, 2*n, 4*n] # These differ from the version for normal because in normal, n represented size for one cond rather than overall size
    
    if len(sys.argv) > 7 and sys.argv[7].startswith("forceActions"):
        FORCE_ACTIONS = True
        num_to_force = float(sys.argv[7].split(",")[1])
    else:
        num_to_force = 0
    
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
        try:
            softmax_beta = float(sys.argv[7])
            reordering_fn = reorder_samples_in_rewards.order_by_named_column('Action1OracleActualReward')
            if len(sys.argv) > 8:
                reordering_fn_specifier = sys.argv[8]
                reordering_fn = reorder_samples_in_rewards.get_reordering_fn(reordering_fn_specifier)
        except:
            print("Parsing error:", sys.exc_info()[0])# different kind of argument
             
            
    num_samples_before_switch = -1
    if len(sys.argv) > 8 and sys.argv[8].startswith("numSamples:"):
        num_samples_array = sys.argv[8].split(":")[1:]
        num_samples_before_switch = int(num_samples_array[0])
        num_samples_after_switch = int(num_samples_array[1])

    if len(sys.argv) > 9 and sys.argv[9].startswith("switchIfNonSig:"):
        switch_to_best_if_nonsignificant = sys.argv[9].split(":")[1].lower() == "true"
    else:
        switch_to_best_if_nonsignificant = False

    # n here is what's required for .8 power (number in both conditions)
    step_sizes_before_switch = [int(round(0.25*n)), int(round(0.5*n)), n]#, 2*n]
    if len(sys.argv) > 10 and sys.argv[10].startswith("multiplier:"):
        multiplier = int(sys.argv[10].split(":")[1])
        print("multiplier:", multiplier)
    else:
        multiplier = 5
    step_sizes = [(multiplier+1)*step_size for step_size in step_sizes_before_switch]
    prior_params = None
    if recalculate_bandits:
            
        if bandit_type == "uniform":
            if num_samples_before_switch > 0:
                step_sizes = [num_samples_before_switch + num_samples_after_switch]
                run_simulations_uniform_random_binary(num_sims, prob_per_arm, 
                                                      num_samples_before_switch, num_samples_after_switch, 
                                                      outfile_directory, forceActions = num_to_force, 
                                                      switch_to_best_if_nonsignificant=switch_to_best_if_nonsignificant)
            else:
                for num_steps in step_sizes_before_switch:
                    run_simulations_uniform_random_binary(num_sims, prob_per_arm, 
                                                          num_steps, num_steps*multiplier,
                                                          outfile_directory, forceActions = num_to_force, 
                                                          switch_to_best_if_nonsignificant=switch_to_best_if_nonsignificant)
        else:
            if len(sys.argv) > 5:
                if sys.argv[5] == "armsHigh":
                    # Arms should be higher than the prior
                    priorProportionOnSuccess = min(prob_per_arm)*run_effect_size_simulations_beta.PRIOR_PROPORTION_DIFFERENCE
                elif sys.argv[5] == "armsLow":
                    # Arms should be lower than the prior
                    priorProportionOnSuccess = 1 - (1-max(prob_per_arm))*run_effect_size_simulations_beta.PRIOR_PROPORTION_DIFFERENCE
                else:
                    # Prior should be uniform (in between arms)
                    priorProportionOnSuccess = .5
                # Make sure the prior sums to 2, mirroring the successes/failures of uniform prior
                prior_params = [priorProportionOnSuccess*2, 2-priorProportionOnSuccess*2]
                print("Prior params: ", prior_params)
                
                run_effect_size_simulations_beta.run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory, 
                                prior_params[0], prior_params[1], 
                                softmax_beta = softmax_beta, reordering_fn = reordering_fn,
                                forceActions = num_to_force)
            else:
                run_effect_size_simulations_beta.run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory, forceActions = num_to_force)
    
    outfile_prefix = outfile_directory  + bandit_type_prefix + str(effect_size);
    if effect_size == 0:
        # Then include the n  in the prefix
        outfile_prefix += "N" + str(n) 
    df = calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, 
                                        switch_to_best_if_nonsignificant, step_sizes_before_switch, 
                                        DESIRED_ALPHA, is_binary=True)

    df.to_pickle(outfile_prefix + 'Df.pkl')
    df_by_trial = calculate_by_trial_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)

#     df_by_trial.to_pickle(outfile_prefix + 'DfByTrial.pkl')
#     # Print various stats
#     summary_text = effect_size_sim_output_viz.print_output_stats(df, prob_per_arm, False, prior_params = prior_params, reordering_info = softmax_beta)
#     with open(outfile_prefix + 'SummaryText.txt', 'w', newline='') as outf:
#         outf.write(summary_text)
#     overall_stats_df = effect_size_sim_output_viz.make_overall_stats_df(df, prob_per_arm, False, effect_size)
#     overall_stats_df.to_pickle(outfile_prefix + 'OverallStatsDf.pkl')
    
    
def mainNormalReward():
    print("Running normal")

    recalculate_bandits = True
    mean1, mu1 = get_mean_and_prior_from_string(sys.argv[1])
    mean2, mu2 = get_mean_and_prior_from_string(sys.argv[2])
    if mu1 != mu2 and mu2 != 0:
        print("Error: different priors on the arms aren't implemented for normal bandits.")
        exit()
        
    means = [mean1, mean2]
    if mean1 == mean2:
        # effect size must be 0 - interpret third argument as the variance to use for the arms
        variance = float(sys.argv[3])
        # n, for basing number of steps off of, also has to be set
        n = int(sys.argv[7])
        # equal arm means indicates there's no effect
        effect_size = 0
    elif len(sys.argv) > 8 and sys.argv[8].startswith("fixedVariance"):
        # We're running a simulation based on an existing experiment, so we want to set the variance
        # manually
        variance = [float(num) for num in sys.argv[3].split(",")]
        # n, for basing number of steps off of, also has to be set
        n = int(sys.argv[7])
        # equal arm means indicates there's no effect
        effect_size = 0 #TODO: fix!
    else:
        effect_size = float(sys.argv[3])
        variance = get_var_from_effect_size(mean1, mean2, effect_size)
        nobs1 = statsmodels.stats.power.tt_ind_solve_power(effect_size, None, DESIRED_ALPHA, DESIRED_POWER, 1)
        n = math.ceil(nobs1)
    num_sims = int(sys.argv[4])
    outfile_directory = sys.argv[5]
    bandit_type = "Thompson"
    bandit_type_prefix = 'NG'
    if len(sys.argv) > 6:
        bandit_type = sys.argv[6]
    if bandit_type == "uniform":
        bandit_type_prefix = "NU"# Normal rewards, uniform policy
    
    if len(sys.argv) > 8 and sys.argv[8].startswith("forceActions"):
        FORCE_ACTIONS = True
        num_to_force = float(sys.argv[8].split(",")[1])
    else:
        num_to_force = 0
    reorder_rewards = False
    softmax_beta = None
    num_samples_before_switch = -1
    if len(sys.argv) > 8 and sys.argv[8].startswith("numSamples:"):
        num_samples_array = sys.argv[8].split(":")[1:]
        num_samples_before_switch = int(num_samples_array[0])
        num_samples_after_switch = int(num_samples_array[1])

    if len(sys.argv) > 9 and sys.argv[9].startswith("switchIfNonSig:"):
        switch_to_best_if_nonsignificant = sys.argv[9].split(":")[1].lower() == "true"
    else:
        switch_to_best_if_nonsignificant = False
        

    num_arms = 2

    # n here is half of what's required for .8 power (number in one condition
    step_sizes_before_switch = [int(round(0.5*n)), n, 2*n, 4*n]
    if len(sys.argv) > 10 and sys.argv[10].startswith("multiplier:"):
        multiplier = int(sys.argv[10].split(":")[1])
        print("multiplier:", multiplier)
    else:
        multiplier = 5
    step_sizes = [(multiplier+1)*step_size for step_size in step_sizes_before_switch]

    if recalculate_bandits:
        if bandit_type == "uniform":
            if num_samples_before_switch > 0:
                step_sizes = [num_samples_before_switch + num_samples_after_switch]
                run_simulations_uniform_random(num_sims, means, variance, num_samples_before_switch, 
                                               num_samples_after_switch, 
                                               outfile_directory, forceActions = num_to_force, 
                                               switch_to_best_if_nonsignificant= switch_to_best_if_nonsignificant)
            else:
                for num_steps in step_sizes_before_switch:
                    run_simulations_uniform_random(num_sims, means, variance, num_steps, 
                                                   num_steps*multiplier, 
                                                   outfile_directory, forceActions = num_to_force, 
                                                   switch_to_best_if_nonsignificant= switch_to_best_if_nonsignificant)
        else:
            if reorder_rewards:
                print("Error: reward reordering not implemented for run switch to best simulations")
            else:
                if num_samples_before_switch > 0:
                    step_sizes = [num_samples_before_switch + num_samples_after_switch]
                run_simulations(num_sims, means, variance, step_sizes, outfile_directory, prior_mean = mu1, forceActions = num_to_force)

    outfile_prefix = outfile_directory  + bandit_type_prefix + str(effect_size);
    if effect_size == 0:
        # Then include the n and the arm variance in the prefix
        outfile_prefix += "N" + str(n) + "Var" + str(variance)
    print("step_sizes:",step_sizes)
    df = calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, 
                                        switch_to_best_if_nonsignificant, step_sizes_before_switch, DESIRED_ALPHA)
    df.to_pickle(outfile_prefix + 'Df.pkl')
    df_by_trial = calculate_by_trial_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)
    df_by_trial.to_pickle(outfile_prefix + 'DfByTrial.pkl')

    # Print various stats
    summary_text = effect_size_sim_output_viz.print_output_stats(df, means + [variance], True, effect_size, reordering_info = softmax_beta)
 
    with open(outfile_prefix + 'SummaryText.txt', 'w', newline='') as outf:
        outf.write(summary_text)
    overall_stats_df = effect_size_sim_output_viz.make_overall_stats_df(df, means + [variance], True, effect_size)
    overall_stats_df.to_pickle(outfile_prefix + 'OverallStatsDf.pkl')
         
    
         
     
if __name__=="__main__":
    if sys.argv[-1] == "binary":
        mainBinaryRewards()
    else:
        mainNormalReward()
