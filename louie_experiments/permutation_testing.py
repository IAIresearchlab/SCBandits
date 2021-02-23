'''
Created on Aug 10, 2018

@author: rafferty
'''
import sys
import csv
import beta_bernoulli
import ng_normal
import numpy as np
from output_format import H_ALGO_OBSERVED_REWARD, H_ALGO_ACTION
import pandas as pd
import read_config
import run_effect_size_simulations_beta
import run_effect_size_simulations
from read_config import NUM_PERMUTATIONS_HEADER
from contextlib import contextmanager
from forced_actions import forced_actions

NUM_STEPS_HEADER = "NumSteps"
NUM_SAMPLES_BY_ACTION_HEADER = "NumSamplesAction{}"
PRIOR_MEAN_HEADER = "PriorMean"
SIM_NUMBER_HEADER = "SimNumber"
PVALUE_HEADER = "pValue"
ACTUAL_STAT_HEADER = "ActualStat"
IS_BINARY_HEADER = "IsBinary"
NUM_PERMUTATIONS_OUT_HEADER = "NumPermutations"

PERMUTATION_TEST_OUT_FILE_SUFFIX = "PermutationTestResults.pkl" 
PERMUTATION_TEST_OUT_FILE_CSV_SUFFIX = "PermutationTestResults.csv" 
WRITE_CSV = False

debug = False
num_actions = 2

def test_all_in_directory(num_sims, step_sizes, outfile_directory, outfile_prefix, prior,
                                num_actions = 2,
                                 config = {}):
    assert num_actions == 2
    if config.get(read_config.NUM_STEP_SIZES_HEADER, len(step_sizes)) < len(step_sizes) :
        step_sizes = step_sizes[:config.get(read_config.NUM_STEP_SIZES_HEADER)]
        
    if debug or True:
        print('Config:', config)
    
    

    rows = []
    all_stats_file = config.get(read_config.PRINT_ALL_STATS_FILE_HEADER)
    none_context = contextmanager(lambda: iter([None]))()
    with (open(all_stats_file) if all_stats_file is not None else none_context) as stats_outfile: 
        if stats_outfile is not None:
            csvwriter = csv.writer()
        else:
            csvwriter = None
        for i in range(num_sims):
            for num_steps in step_sizes:
                forced = run_effect_size_simulations.make_forced_actions(num_actions, num_steps, config[read_config.FORCE_ACTIONS_HEADER])

                if config[read_config.BINARY_REWARDS_HEADER]:
                    actions_infile = run_effect_size_simulations_beta.get_output_filename(outfile_directory, num_steps, i)
                else:
                    actions_infile = run_effect_size_simulations.get_output_filename(outfile_directory, num_steps, i)
                    
                if debug:
                    print("processing file:",actions_infile)
                outfile_row = make_outfile_row(actions_infile, config, num_steps, prior, csv_writer_all_stats=csvwriter, forced_actions = forced)
                if debug or True:
                    print("processing completed:",actions_infile)
    
                outfile_row[SIM_NUMBER_HEADER] = i
                rows.append(outfile_row)
    dataframe_headers = [NUM_STEPS_HEADER,SIM_NUMBER_HEADER, PRIOR_MEAN_HEADER,
                         PVALUE_HEADER, ACTUAL_STAT_HEADER]
    for action in range(num_actions):
        dataframe_headers.append(NUM_SAMPLES_BY_ACTION_HEADER.format(action + 1))
    df = pd.DataFrame.from_records(rows, columns = dataframe_headers)
    df[IS_BINARY_HEADER] = config[read_config.BINARY_REWARDS_HEADER]
    df[NUM_PERMUTATIONS_OUT_HEADER] = config[NUM_PERMUTATIONS_HEADER]
    if debug:
        print("writing to", outfile_prefix + PERMUTATION_TEST_OUT_FILE_SUFFIX) 
    df.to_pickle(outfile_prefix + PERMUTATION_TEST_OUT_FILE_SUFFIX)
    if WRITE_CSV:
        df.to_csv(outfile_prefix + PERMUTATION_TEST_OUT_FILE_CSV_SUFFIX)
                
    
            
def make_outfile_row(actions_infile, config, num_steps, prior, csv_writer_all_stats = None, stat_fn = None, forced_actions = None):
        outfile_row = {}
        actions_df = pd.read_csv(actions_infile,skiprows=1)
        is_binary = config[read_config.BINARY_REWARDS_HEADER]
        if stat_fn == None:
            if is_binary:
                stat_fn = diff_in_est_p_stat
                outfile_row[PRIOR_MEAN_HEADER] = prior[0] / sum(prior)
            else:
                stat_fn = diff_in_est_p_stat # separating these cases to make clear we may want different stats for each
                outfile_row[PRIOR_MEAN_HEADER] = prior[0]
        epsilon = 1 if config.get(read_config.SAMPLING_METHOD_HEADER,'thompson') == 'uniform' else 0
        outfile_row[PVALUE_HEADER], all_stats, outfile_row[ACTUAL_STAT_HEADER] = permutation_test(actions_df, 
                                                                         stat_fn, 
                                                                         prior,
                                                                         is_binary,
                                                                         num_permutations = config[NUM_PERMUTATIONS_HEADER],
                                                                         epsilon = epsilon,
                                                                         forced_actions = forced_actions)
        outfile_row[NUM_STEPS_HEADER] = num_steps
        for action in range(num_actions):
            outfile_row[NUM_SAMPLES_BY_ACTION_HEADER.format(action + 1)] = sum(actions_df.loc[:,H_ALGO_ACTION] == (action + 1))
        
        if csv_writer_all_stats != None:
            csv_writer_all_stats.writerow([num_steps] + all_stats)
        return outfile_row
        
        
def non_parametric_confidence_interval(actions_df, stat_fn, prior, is_binary = True, num_permutations = 5, epsilon = 0, ci_size = .95, grid_size = .05, forced_actions = None):
    in_ci = []
    non_offset_tau_0 = 0
    for grid_offset in np.arange(-3,3.001, grid_size):
        tau_0 = non_offset_tau_0 + grid_offset
        
        rewards = actions_df.loc[:,H_ALGO_OBSERVED_REWARD];
        original_actions = actions_df.loc[:,H_ALGO_ACTION]
        rewards_mod = rewards.copy()
        rewards_mod.loc[original_actions == 1] = rewards_mod.loc[original_actions == 1] - tau_0
        actual_stat = stat_fn(original_actions, rewards_mod)
    
        all_stats = []
        more_extreme_count = 0
        for i in range(num_permutations):
            if is_binary:
                models = [beta_bernoulli.BetaBern(prior[0], prior[1]) for _ in range(num_actions)]
            else:
                models = [ng_normal.NGNormal(mu=prior[0], k=prior[1], alpha=prior[2], beta=prior[3]) for _ in range(num_actions)]
    
    
            chosen_actions, models = calculate_thompson_single_bandit_permutation_testing(rewards, models, epsilon = epsilon, forced_actions=forced_actions)
            cur_stat = stat_fn(chosen_actions, rewards_mod)
            if cur_stat >= actual_stat:
                more_extreme_count += 1
            all_stats.append(cur_stat)
            if debug and (i % 100) == 0:
                print(i,"/ num_permutations:", more_extreme_count)
        pvalue = more_extreme_count / num_permutations
        if np.isnan(actual_stat):
            pvalue = np.nan
        if (1-pvalue) <= ci_size:
            in_ci.append(tau_0)
            
    return in_ci   
    
def permutation_test(actions_df, stat_fn, prior, is_binary = True, num_permutations = 5, epsilon = 0, forced_actions = None):
    rewards = actions_df.loc[:,H_ALGO_OBSERVED_REWARD]; #"ObservedRewardofAction"
    original_actions = actions_df.loc[:,H_ALGO_ACTION] #"AlgorithmAction"
    actual_stat = stat_fn(original_actions, rewards)

    all_stats = []
    more_extreme_count = 0
    for i in range(num_permutations):
        if is_binary:
            models = [beta_bernoulli.BetaBern(prior[0], prior[1]) for _ in range(num_actions)]
        else:
            models = [ng_normal.NGNormal(mu=prior[0], k=prior[1], alpha=prior[2], beta=prior[3]) for _ in range(num_actions)]


        chosen_actions, models = calculate_thompson_single_bandit_permutation_testing(rewards, models, epsilon = epsilon, forced_actions=forced_actions)
        cur_stat = stat_fn(chosen_actions, rewards)
        if cur_stat >= actual_stat:
            more_extreme_count += 1
        all_stats.append(cur_stat)
        if debug and (i % 100) == 0:
            print(i,"/ num_permutations:", more_extreme_count)
    pvalue = more_extreme_count / num_permutations
    if np.isnan(actual_stat):
        pvalue = np.nan
    return pvalue, all_stats, actual_stat
    
def diff_in_est_p_stat(actions, rewards):
    phats = []
    actions = np.array(actions)
    for action in range(num_actions):
        phat = np.mean(rewards[actions == (action + 1)])
        phats.append(phat)
    return abs(phats[1] - phats[0])
    

def calculate_thompson_single_bandit_permutation_testing(rewards, models, epsilon = 0, forced_actions = None):
    '''
    Performs a version of Thompson sampling for permutation hypothesis test. rewards is an iterable
    that contains the actual observed rewards from an experiment/simulation. These will be used
    as the rewards regardless of what actions we actually choose. We update the model that 
    corresponds to the action that was chosen in this iteration of Thompson sampling, and
    we use the reward that was originally observed. This is because we're testing against the
    hypothesis that condition doesn't matter - i.e., reward distributions are the same across
    action choices.
    epsilon is used to allow uniform action sampling to be done, or thompson with extra suboptimality.
    At every step there is epsilon probability of choosing an action uniformly at random.
    Returns a list of the actions chosen this time around, which will be used in conjunction
    with the actual rewards to calculate the test statistic.
    '''
    chosen_actions = []
    for i,reward in zip(range(len(rewards)), rewards):
        # no context
        context = None
        # choose an action
        if forced_actions is not None and i < len(forced_actions.actions): 
            action = forced_actions.actions[i]
        elif np.random.random() < epsilon:
            action = np.random.randint(len(models))
        else:
            samples = [models[a].draw_expected_value(context) for a in range(len(models))] 
            action = np.argmax(samples)#sample from posterior and choose as in TS
        chosen_actions.append(action + 1)

        # reward is whatever the reward is for the action that was chosen originally -
        # i.e., it's not dependent on the action choice
        # update posterior distribution with observed reward, attributing it to whatever action was
        # chosen
        models[action].update_posterior(context, reward)
    return chosen_actions, models

def main():
    config = read_config.get_json_arguments(sys.argv[1])
    actions_df = pd.read_csv(config["actions_infile"],skiprows=1)

    pvalue, all_stats, actual_stat = permutation_test(actions_df, diff_in_est_p_stat, config["prior"], 
                                                      is_binary = config[read_config.BINARY_REWARDS_HEADER], 
                                                      num_permutations = config[read_config.NUM_PERMUTATIONS_HEADER])
    print("p =", pvalue)
    print("actual stat:", actual_stat)
    print("all stats:", all_stats)
    in_ci = non_parametric_confidence_interval(actions_df, diff_in_est_p_stat, config["prior"], 
                                                  is_binary = config[read_config.BINARY_REWARDS_HEADER], 
                                                  num_permutations = config[read_config.NUM_PERMUTATIONS_HEADER])
    print(in_ci)
if __name__ == "__main__":
    main()
    