'''
Created on Jul 21, 2017

@author: arafferty
'''
import argparse 
import math
import random
import sys

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
from epsilon_greedy import *


DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

PRIOR_PROPORTION_DIFFERENCE = .5 # When making arms that don't have the prior in the center what proportion of the remaining space to use for the arms

action_header ='AlgorithmAction'
obs_reward_header = 'ObservedRewardofAction'

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
    success_template = 'Action{0}SuccessCount'
    failure_template =  'Action{0}FailureCount'
    successes = [success_template.format(i) for i in range(1, num_arms + 1)]
    failures = [failure_template.format(i) for i in range(1, num_arms + 1)]

    arm_mean_columns = ['mean_arm_' + str(i) for i in range(1, num_arms + 1)]
    arm_var_columns = ['var_arm_' + str(i) for i in range(1, num_arms + 1)]
    
    all_dfs = []
    #print("ARMSTATS---")
    #print("num_steps", num_steps)
    for i in range(num_sims):
        output_file = pd.read_csv(get_output_filename(outfile_directory, num_steps, i),skiprows=1)
       # print("output_file", output_file)
        mean_dfs = [output_file[successHeader] / (output_file[successHeader] + output_file[failureHeader]) for successHeader, failureHeader in zip(successes, failures)]
        all_means_df = pd.DataFrame({arm_mean_column: col for arm_mean_column, col in zip(arm_mean_columns,mean_dfs)})
        var_dfs = [scipy.stats.beta.var(output_file[successHeader], output_file[failureHeader]) for successHeader, failureHeader in zip(successes, failures)]
        all_vars_df = pd.DataFrame({arm_var_column: col for arm_var_column, col in zip(arm_var_columns,var_dfs)})
        means_and_var_df = pd.concat([all_means_df,all_vars_df], axis=1, join="inner")
        means_and_var_df.insert(0,'num_steps',num_steps)
        means_and_var_df.insert(0,'sim',i)
        means_and_var_df.insert(0,'trial',range(0,  all_vars_df.shape[0]))
        all_dfs.append(means_and_var_df)

    # Now do some aggregating
    #print("all_dfs len", len(all_dfs))

    df = all_dfs[0].append(all_dfs[1:])
    return df

def make_stats_row_from_df(cur_data, include_power, effect_size = None, alpha = None):
    '''Calculates output statistics given the data frame cur_data. If include_power, includes the power calculation.
    efffect_size and alpha are only required/used if power is calculated
    '''
    sample_sizes = np.array([np.sum(cur_data[action_header] == i) for i in range(1,3)])
    successes = np.array([np.sum(cur_data[cur_data[action_header] == 1][obs_reward_header]), 
             np.sum(cur_data[cur_data[action_header] == 2][obs_reward_header])])
    #calculate sample size and mean
    cur_row = {}
   
    sample_size_1 = sample_sizes[0]
    sample_size_2 = sample_sizes[1]
    cur_row['sample_size_1'] = sample_size_1
    cur_row['sample_size_2'] = sample_size_2

    mean_1 = np.mean(cur_data[cur_data[action_header] == 1][obs_reward_header])# JN mean for arm 1
    mean_2 = np.mean(cur_data[cur_data[action_header] == 2][obs_reward_header])#JN mean for arm 2

    cur_row['mean_1'] = mean_1
    cur_row['mean_2'] = mean_2
    #SE = sqrt[(P^hat_A*(1-P^hat_A)/N_A + (P^hat_B*(1-P^hat_B)/N_B]
    SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2)
    wald_type_stat = (mean_1 - mean_2)/SE #(P^hat_A - P^hat_b)/SE
    #print('wald_type_stat:', wald_type_stat)
    wald_pval = (1 - scipy.stats.norm.cdf(np.abs(wald_type_stat)))*2 #Two sided, symetric, so compare to 0.05

    cur_row['wald_type_stat'] = wald_type_stat #TODO add in Delta!!!!
    cur_row['wald_pval'] = wald_pval
    #print("wald_pval", wald_pval)
    #calculate total reward
    cur_row['total_reward'] = np.sum(cur_data[obs_reward_header])
     
    #calculate power
    cur_row['ratio'] = sample_sizes[0] / sample_sizes[1]
    if include_power:
        cur_row['power'] = smp.GofChisquarePower().solve_power(effect_size, nobs = sum(sample_sizes), n_bins=(2-1)*(2-1) + 1, alpha = alpha)
    cur_row['actual_es'] = calculate_effect_size(sample_sizes, successes)
    

    #calculate chi squared contingency test
    table = sms.Table(np.stack((successes,sample_sizes - successes)).T)
    rslt = table.test_nominal_association()
    cur_row['stat'] = rslt.statistic
    cur_row['pvalue'] = rslt.pvalue
    cur_row['df'] = rslt.df
    # Added to match normal rewards
    cur_row['statUnequalVar'],cur_row['pvalueUnequalVar'], cur_row['dfUnequalVar'] = cur_row['stat'],cur_row['pvalue'],cur_row['df']
    return cur_row

def calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, alpha = 0.05):
    '''
    - Output that we'll look at:
    -- Given the actual average ratios of conditions, what is our average power (calculated using formulas above)?
    -- Actually conducting a t-test on our observations, what are our average statistics and p-values? Calculate the rate of rejecting the null hypothesis for the simulations (we'd expect it to be .8, since that's what power means).
    -- How large is our actual effect size on average, given that we have unequal samples?
    '''
    # Go through each file to collect the vectors of rewards we observed from each arm [i.e. condition]
    rows =[]
    for num_steps in step_sizes: 
        for i in range(num_sims):
            output_file = pd.read_csv( get_output_filename(outfile_directory, num_steps, i),skiprows=1)
            cur_row = make_stats_row_from_df(output_file, True, effect_size, alpha)
            cur_row['num_steps'] = num_steps
            cur_row['sim'] = i
                                                                   
            rows.append(cur_row)
    # Now do some aggregating
    df = pd.DataFrame(rows, columns=['num_steps', 'sim', 'sample_size_1', 'sample_size_2', 'mean_1','mean_2',\
     'total_reward', 'ratio', 'power', 'actual_es', 'stat', 'pvalue', \
     'df','statUnequalVar','pvalueUnequalVar', 'dfUnequalVar', 'wald_pval', 'wald_type_stat'])
    
    return df



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
    
    

def get_rewards_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the rewards for sim_num that has num_steps steps.
    '''
    separator = '/'
    if outfile_directory.endswith('/'):
        separator = ''
    reward_data_file = outfile_directory  + separator + 'tbb_rewards_{0}_{1}.csv'
    return reward_data_file.format(num_steps, sim_num)

def get_reordered_rewards_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the rewards for sim_num that has num_steps steps.
    '''
    separator = '/'
    if outfile_directory.endswith('/'):
        separator = ''
    reward_data_file = outfile_directory + separator +'/tbb_rewards_reordered_{0}_{1}.csv'
    return reward_data_file.format(num_steps, sim_num)

def get_output_filename(outfile_directory, num_steps, sim_num):
    '''
    Returns the name of the file that will have the actions taken for sim_num that has num_steps steps.
    '''
    separator = '/'
    if outfile_directory.endswith('/'):
        separator = ''
    results_data_file = outfile_directory + separator + 'tbb_actions_{0}_{1}.csv'
    return results_data_file.format(num_steps, sim_num)

def run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory, successPrior = 1, failurePrior = 1, softmax_beta = None, \
    reordering_fn = None, forceActions = 0, batch_size = 1, burn_in_size = 1, epsilon = 0.1):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy.
    '''

    for i in range(num_sims):
      #  num_steps_prev = 0
        for num_steps in step_sizes:
            if forceActions != 0:
#                 print("Forcing actions:", forceActions)
                forced = run_effect_size_simulations.make_forced_actions(len(prob_per_arm), num_steps, forceActions)
            else:
                forced = forced_actions()
            cur_reward_file = get_rewards_filename(outfile_directory, num_steps, i)
            generate_single_bandit.generate_file(np.array(prob_per_arm),
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
           # models = [beta_bernoulli.BetaBern(success=successPrior, failure=failurePrior) for _ in range(len(prob_per_arm))]

            #Don't pass model, then will be Greedy
            calculate_epsilon_single_bandit(reordered_reward_file, 
                                         num_actions=len(prob_per_arm), 
                                         dest= cur_output_file,
                                         forced = forced, epsilon = epsilon)



           # num_steps_prev = num_steps
            
def run_simulations_empirical_rewards(num_sims, reward_file, experiment_id, reward_header, is_cost, 
                                      outfile_directory, successPrior = 1, failurePrior = 1,  forceActions = 0,
                                      shuffle_data = False):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy.
    '''
    num_actions = 2
    max_steps = -1
    means = []
    variance = []
    for i in range(num_sims):
        arm_1_rewards, arm_2_rewards = get_assistments_rewards.read_assistments_rewards(reward_file, 
                                                                                        reward_header, 
                                                                                        experiment_id,
                                                                                        is_cost)
        if shuffle_data:
            random.shuffle(arm_1_rewards)
            random.shuffle(arm_2_rewards)
        max_steps = len(arm_1_rewards) + len(arm_2_rewards)
        means = [np.mean(arm_1_rewards), np.mean(arm_2_rewards)]
        variance= [np.var(arm_1_rewards), np.var(arm_2_rewards)]
        if forceActions != 0:
            print("Forcing actions:", forceActions)
            forced = run_effect_size_simulations.make_forced_actions(num_actions, len(arm_1_rewards) + len(arm_2_rewards), forceActions)
        else:
            forced = forced_actions()

        
        cur_output_file = get_output_filename(outfile_directory, len(arm_1_rewards) + len(arm_2_rewards), i)
        models = [beta_bernoulli.BetaBern(success=successPrior, failure=failurePrior) for _ in range(num_actions)]
        thompson_policy.calculate_thompson_single_bandit_empirical_params(arm_1_rewards,
                                     arm_2_rewards, 
                                     num_actions=num_actions, 
                                     dest= cur_output_file, 
                                     models=models, 
                                     action_mode=thompson_policy.ActionSelectionMode.prob_is_best, 
                                     relearn=True,
                                     forced = forced)
    return max_steps, means, variance

            
def run_simulations_uniform_random(num_sims, prob_per_arm, step_sizes, outfile_directory, forceActions = 0):
    '''
    Runs num_sims bandit simulations with several different sample sizes (those in the list step_sizes). 
    Bandit uses the thompson_ng sampling policy.
    '''

    for i in range(num_sims):
        for num_steps in step_sizes:
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
            thompson_policy.calculate_thompson_single_bandit(cur_reward_file, 
                                         num_actions=len(prob_per_arm), 
                                         dest= cur_output_file, 
                                         models=models, 
                                         action_mode=thompson_policy.ActionSelectionMode.prob_is_best,
                                         epsilon = 1.0, 
                                         relearn=True,
                                         forced = forced)
    

def get_prob_per_arm_from_effect_size(effect_size, center = 0.5):
    '''
    Calculates the probability of success on each arm that would be needed to get the given effect_size.
    Assumes the sample sizes will be equal, and that we have two arms, one with probability .5 + x, the other
    with probability .5 - x, for some x.
    '''
    x = effect_size / 2
    return [center + x, center - x]

# def getParsedArguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--numSims", type=int, default=1, help="total number of simulations of the bandit to run (each simulation involves multiple steps)")
#     parser.add_argument("-o", "--outfile_dir", type=str, 
#                         help="directory where all of the output will be written")
#     parser.add_argument("-e", "--effectSize", type=float,
#                         help="set the arms to have this effect size (must be > 0)")
#     parser.add_argument("--armProbs", type=float,
#                         help="set the arms to have this effect size (must be > 0)")
#     parser.add_argument("-w", "--writeAllData", type=bool, default=True,
#                         help="if true, write all runs to the table file; otherwise, write only averages")
#     parser.add_argument("--twoFactor", type=bool, default=False,
#                         help="if true, uses two-factor latent structure reward; otherwise, uses MVN")
# 
#     args = parser.parse_args()
#     return args

def main():
    recalculate_bandits = True
    outfile_directory = sys.argv[3]
    #batch_size = 1.0
 #   if len(sys.argv) > 5:
  #      epsilon = float(sys.argv[5])
   #     print("epsilon", epsilon)
    if "epsilon" in outfile_directory:
        print(outfile_directory)
        epsilon = float(outfile_directory.split("epsilon")[-1].split("/")[0].strip("="))
        print("epsilon", epsilon)
    
    num_sims = int(sys.argv[2])

    burn_in_size, batch_size = int(outfile_directory.split("=")[-1].split('-')[0]), int(outfile_directory.split("=")[-1].split('-')[1])
    print("burn_in_size, batch_size", burn_in_size, batch_size)
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
        effect_size = float(sys.argv[1].split("-")[0])
        center = float(sys.argv[1].split("-")[1])
        prob_per_arm = get_prob_per_arm_from_effect_size(effect_size, center)
        # Assumes we have two arms
        nobs_total = smp.GofChisquarePower().solve_power(effect_size, n_bins=(2-1)*(2-1) + 1, alpha = DESIRED_ALPHA, power = DESIRED_POWER)
#         print("Calculated nobs for effect size:", nobs_total)
        n = math.ceil(nobs_total)
    #step_sizes = [math.ceil(n/2), n, 2*n] # These differ from the version for normal because in normal, n represented size for one cond rather than overall size
    step_sizes = [math.ceil(n/2), n, 2*n, 4*n] # These differ from the version for normal because in normal, n represented size for one cond rather than overall size

    print("prob_per_arm", prob_per_arm)
    if len(sys.argv) > 7 and sys.argv[7].startswith("forceActions"):
        run_effect_size_simulations.FORCE_ACTIONS = True
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
        softmax_beta = float(sys.argv[7])
        reordering_fn = reorder_samples_in_rewards.order_by_named_column('Action1OracleActualReward')
        if len(sys.argv) > 8:
            reordering_fn_specifier = sys.argv[8]
            reordering_fn = reorder_samples_in_rewards.get_reordering_fn(reordering_fn_specifier)

    prior_params = None
    if recalculate_bandits:
            
        if bandit_type == "uniform":
            run_simulations_uniform_random(num_sims, prob_per_arm, step_sizes, outfile_directory, forceActions = num_to_force)
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
                
                run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory, 
                                prior_params[0], prior_params[1], 
                                softmax_beta = softmax_beta, reordering_fn = reordering_fn,
                                forceActions = num_to_force, batch_size = batch_size, burn_in_size = burn_in_size, epsilon = epsilon)
            else:
                run_simulations(num_sims, prob_per_arm, step_sizes, outfile_directory, forceActions = num_to_force, batch_size = batch_size, \
                    burn_in_size = burn_in_size, epsilon = epsilon)
    
    outfile_prefix = outfile_directory  + bandit_type_prefix + str(effect_size);
    if effect_size == 0:
        # Then include the n  in the prefix
        outfile_prefix += "N" + str(n) 

    df = calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)
    df.to_pickle(outfile_prefix + 'Df.pkl')
    df_by_trial = calculate_by_trial_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)

    df_by_trial.to_pickle(outfile_prefix + 'DfByTrial.pkl')
    # Print various stats
    summary_text = effect_size_sim_output_viz.print_output_stats(df, prob_per_arm, False, prior_params = prior_params, reordering_info = softmax_beta)
    with open(outfile_prefix + 'SummaryText.txt', 'w', newline='') as outf:
        outf.write(summary_text)
    overall_stats_df = effect_size_sim_output_viz.make_overall_stats_df(df, prob_per_arm, False, effect_size)
    overall_stats_df.to_pickle(outfile_prefix + 'OverallStatsDf.pkl')
     
    # Make histogram
    hist_figure = effect_size_sim_output_viz.make_hist_of_trials(df)
    hist_figure.savefig(outfile_prefix + 'HistOfConditionProportions.pdf', bbox_inches='tight')
 
    # Make line plot
    test_stat_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'stat')
    test_stat_figure.savefig(outfile_prefix + 'TestStatOverTime.pdf', bbox_inches='tight')
     
    pvalue_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'pvalue')
    pvalue_figure.savefig(outfile_prefix + 'PValueOverTime.pdf', bbox_inches='tight')
     
    # Plot power
    power_figure = effect_size_sim_output_viz.plot_power_by_steps(df_by_trial, DESIRED_ALPHA, DESIRED_POWER)
    power_figure.savefig(outfile_prefix + 'PowerOverTime.pdf', bbox_inches='tight')
     
    #Plot reward
    reward_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'total_reward')
    reward_figure = effect_size_sim_output_viz.add_expected_reward_to_figure(reward_figure, prob_per_arm, step_sizes)
    reward_figure.savefig(outfile_prefix + 'RewardOverTime.pdf', bbox_inches='tight')
     
    # Plot arm statistics
    arm_df_by_trial = create_arm_stats_by_step(outfile_directory, num_sims, step_sizes[-1], num_arms)
    arm_stats_figure = effect_size_sim_output_viz.make_by_trial_arm_statistics(arm_df_by_trial, num_arms)
    arm_stats_figure.savefig(outfile_prefix + 'ArmStats.pdf', bbox_inches='tight')
    
def empirical_main():
    # Assumes sys.argv[1] == 'empirical'
    recalculate_bandits = True
    num_arms = 2

    num_sims = int(sys.argv[2])
    
    reward_file = sys.argv[3]
    experiment_id = sys.argv[4]
    reward_header = sys.argv[5]
    
    if sys.argv[6] == "use_cost":
        is_cost = True
    else:
        is_cost = False
    
    outfile_directory = sys.argv[7]
    
    priorProportionOnSuccess = float(sys.argv[8])
    
    forceActions = 0
    
    shuffle_data = False
    if len(sys.argv) > 9:
        shuffle_data = sys.argv[9] == 'True'
    
    bandit_type = "Thompson"
    bandit_type_prefix = 'BB'

    prior_params = None
    if recalculate_bandits:
        # Make sure the prior sums to 2, mirroring the successes/failures of uniform prior
        prior_params = [priorProportionOnSuccess*2, 2-priorProportionOnSuccess*2]
        print("Prior params: ", prior_params)
        
        max_steps, prob_per_arm, variance = run_simulations_empirical_rewards(num_sims, reward_file, experiment_id, reward_header, is_cost, 
                                          outfile_directory, prior_params[0], prior_params[1], forceActions, shuffle_data)
    
    outfile_prefix = outfile_directory  + bandit_type_prefix + experiment_id + reward_header
    effect_size = 0
    step_sizes = [max_steps]
    df = calculate_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)
    df.to_pickle(outfile_prefix + 'Df.pkl')
    df_by_trial = calculate_by_trial_statistics_from_sims(outfile_directory, num_sims, step_sizes, effect_size, DESIRED_ALPHA)

    df_by_trial.to_pickle(outfile_prefix + 'DfByTrial.pkl')
    # Print various stats
    summary_text = effect_size_sim_output_viz.print_output_stats(df, prob_per_arm, False, prior_params = prior_params, reordering_info = 0)
    with open(outfile_prefix + 'SummaryText.txt', 'w', newline='') as outf:
        outf.write(summary_text)
    overall_stats_df = effect_size_sim_output_viz.make_overall_stats_df(df, prob_per_arm, False, effect_size)
    overall_stats_df.to_pickle(outfile_prefix + 'OverallStatsDf.pkl')
     
    # Make histogram
    hist_figure = effect_size_sim_output_viz.make_hist_of_trials(df)
    hist_figure.savefig(outfile_prefix + 'HistOfConditionProportions.pdf', bbox_inches='tight')
 
    # Make line plot
    test_stat_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'stat')
    test_stat_figure.savefig(outfile_prefix + 'TestStatOverTime.pdf', bbox_inches='tight')
     
    pvalue_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'pvalue')
    pvalue_figure.savefig(outfile_prefix + 'PValueOverTime.pdf', bbox_inches='tight')
     
    # Plot power
    power_figure = effect_size_sim_output_viz.plot_power_by_steps(df_by_trial, DESIRED_ALPHA, DESIRED_POWER)
    power_figure.savefig(outfile_prefix + 'PowerOverTime.pdf', bbox_inches='tight')
     
    #Plot reward
    reward_figure = effect_size_sim_output_viz.make_by_trial_graph_of_column(df_by_trial, 'total_reward')
    reward_figure = effect_size_sim_output_viz.add_expected_reward_to_figure(reward_figure, prob_per_arm, step_sizes)
    reward_figure.savefig(outfile_prefix + 'RewardOverTime.pdf', bbox_inches='tight')
     
    # Plot arm statistics
    arm_df_by_trial = create_arm_stats_by_step(outfile_directory, num_sims, step_sizes[-1], num_arms)
    arm_stats_figure = effect_size_sim_output_viz.make_by_trial_arm_statistics(arm_df_by_trial, num_arms)
    arm_stats_figure.savefig(outfile_prefix + 'ArmStats.pdf', bbox_inches='tight')
    
if __name__=="__main__":
    print("HERE 1")
    if sys.argv[1] == 'empirical':
        empirical_main()
    else:
        main()
