'''
Created on May 9, 2018

Takes already run simulations and approximates the probability that one mean is greater than another as well as
the average difference in values from the posteriors based on the final posteriors over each condition.

@author: rafferty
'''
import beta_bernoulli 
import ng_normal 
import pandas as pd
import glob
import re
from get_specs_from_filenames import *
import sys
import json
import os
def get_n_and_sim_num(filename, file_prefix):
    pattern = re.compile(file_prefix + '(?P<n>\d+)_(?P<simNum>\d+).csv')
    search_result = pattern.search(filename)
    if search_result == None:
        print("No n, sim_num found in filename: " + filename)
        return -1, -1# error value
    else:
        return int(search_result.group('n')), int(search_result.group('simNum'))


def simulate_prob_condition1_bigger(models, num_samples = 1000):
    if len(models) != 2:
        print("Error: calc_prob_posteriors_differ defined for comparing two conditions; trying to compare",len(models), "conditions")
    
    count_condition_1_bigger = 0
    diff_condition_1_2 = 0
    for _ in range(num_samples):
        samples = [model.draw_expected_value([]) for model in models]
        if samples[0] > samples[1]:
            count_condition_1_bigger += 1
        diff_condition_1_2 += samples[0] - samples[1]
    
    return count_condition_1_bigger / num_samples, diff_condition_1_2 / num_samples



def get_models_from_simulation(simulation_out_file, is_binary = True):
    df = pd.read_csv(simulation_out_file, header = 1)
    last_row = df.iloc[df.shape[0]-1,:]
    if is_binary:
        # Action1SuccessCount
        models = [beta_bernoulli.BetaBern(last_row.loc['Action' + str(i) + 'SuccessCount'], 
                                          last_row.loc['Action' + str(i) + 'FailureCount'])  for i in range(1,3)]
    else:
        models = [ng_normal.NGNormal(mu = last_row.loc['Action' + str(i) + 'EstimatedMu'], 
                                     k = last_row.loc['Action' + str(i) + 'EstimatedVariance'],
                                     alpha = last_row.loc['Action' + str(i) + 'EstimatedAlpha'],
                                     beta = last_row.loc['Action' + str(i) + 'EstimatedBeta'])  for i in range(1,3)]
    return models


def calculate_averages_for_sim_directory(sim_directory, config):
    reward_type = get_reward_spec_from_filename(sim_directory)
    is_binary = reward_type == 'binary'
    file_prefix = 'tng_actions_'
    if is_binary:
        file_prefix = 'tbb_actions_'
#     sims_for_n = glob.glob(sim_directory + '/' + file_prefix + '*_0.csv')

#     ns = [int(file[(file.find(file_prefix) + len(file_prefix)):file.find('_',file.find(file_prefix) + len(file_prefix))]) for file in sims_for_n]
#     ns.sort()
#     print(ns)

    rows = []
    for file in glob.glob(sim_directory + '/' + file_prefix + '*.csv'):
        n, sim_num = get_n_and_sim_num(file, file_prefix)
        models = get_models_from_simulation(file, is_binary)
        exp_value_difference = models[0].get_expected_value() - models[1].get_expected_value()
        prob_bigger, avg_difference = simulate_prob_condition1_bigger(models)
        rows.append({'sample_size' : n, 'sim_num' : sim_num, 
                     'avg_difference' : avg_difference, 'prob_bigger' : prob_bigger, 
                     'exp_value_difference' : exp_value_difference})
    df = pd.DataFrame(rows)

    pseudo_filename = os.path.join(sim_directory, file_prefix)
    df['reward_type'] = reward_type
    
    prior_type = get_prior_spec_from_filename(pseudo_filename)
    df['prior_type'] = prior_type
    
    sampling_type = get_sampling_spec_from_filename(pseudo_filename)
    df['sampling_type'] = sampling_type


    
    if config["zero_effect_size"]:
        variance = '-1.0'
        if reward_type == 'normal':
            variance = get_variance_from_directoryname(pseudo_filename)
        df['actual_arm_variance'] = float(variance)
    else:
        effect_size = float(get_effect_size_from_directory(sim_directory))
        df['effect_size'] = effect_size
            
            

            
    if config["reordered_rewards"]:
        df['softmax'] = get_softmax_from_filename(pseudo_filename)
        df['reorderingPreference'] = get_preference_type_from_filename(pseudo_filename)
        
    if config["real_data"]:
        df['experimentNum'] = get_experiment_num_from_filename(pseudo_filename)
        df['outcomeMeasure'] = get_outcome_measure_from_filename(pseudo_filename)
        if not config["parameter_only"]:
            df['shuffle'] = get_shuffle_boolean_from_real_data_filename(pseudo_filename)
    print(df.head())
    return df

def load_configuration(configurationFile):
    '''
    Returns the JSON object stored in configuration file.
    Used instead of command line args
    '''
    with open(configurationFile) as jsonFile:
        config = json.load(jsonFile)
    return config

def main():
    config = load_configuration(sys.argv[1])
    # Directories to look in for dataframes
    directories_to_combine = config["directories"]#{'empiricalParametersRedoProblemCount7November2017'}#{'empiricalParametersAll3November2017','empiricalParametersUniform6November2017'}#{'sameArms23August2017', 'sameArms24August2017'}

    out_csv = config["outfile"]
    columns = []
    # Walk through all dataframes to combine to one CSV
    is_first = True # We'll mark the first one we write to use write rather than append and to write headers
    for directory in directories_to_combine:
        for file in os.scandir(directory):
            if file.is_dir() and not file.name.startswith("."):
                #use each directory as a sim directory
                cur_df = calculate_averages_for_sim_directory(file.path, config)
                
                if len(columns) == 0:
                    columns = cur_df.columns
                # Now write to csv in progress
                if is_first:
                    is_first = False
                    cur_df.to_csv(out_csv, encoding='utf-8', index=False, columns = columns)
                else:
                    # After the first file, append and don't write headers again
                    cur_df.to_csv(out_csv, encoding='utf-8', index=False, mode='a', header=False, columns = columns)

def debug():
    calculate_averages_for_sim_directory('/Users/rafferty/banditalgorithms/data/reorderedRewardsSimulations/ngEqualMeansArmsLow1.25/', False)
#     betaModels = []
#     betaModels.append(beta_bernoulli.BetaBern(140,36))
#     betaModels.append(beta_bernoulli.BetaBern(120,56))
#     prob_bigger, avg_difference = simulate_prob_condition1_bigger(betaModels)
#     print("probability bigger:", prob_bigger,"; avg. difference:", avg_difference)
#     
#     ngModels = []
#     ngModels.append(ng_normal.NGNormal(mu=0.5, k=1, alpha=10, beta=20))
#     ngModels.append(ng_normal.NGNormal(mu=0.45, k=1, alpha=10, beta=20))
#     prob_bigger, avg_difference = simulate_prob_condition1_bigger(ngModels)
#     print("probability bigger:", prob_bigger,"; avg. difference:", avg_difference)
#     
#     betaModels = get_models_from_simulation('/Users/rafferty/banditalgorithms/data/reorderedRewardsSimulations/tbb_actions_44_138.csv', True)
#     prob_bigger, avg_difference = simulate_prob_condition1_bigger(betaModels)
#     print("probability bigger:", prob_bigger,"; avg. difference:", avg_difference)
# 
#     normalModels = get_models_from_simulation('/Users/rafferty/banditalgorithms/data/reorderedRewardsSimulations/tng_actions_104_80.csv', False)
#     prob_bigger, avg_difference = simulate_prob_condition1_bigger(normalModels)
#     print("probability bigger:", prob_bigger,"; avg. difference:", avg_difference)

if __name__ == "__main__":
#     debug()
    main()
