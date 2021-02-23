'''
Created on Aug 2, 2018

@author: rafferty
'''
import json

MAX_WEIGHT_HEADER = "max_weight"
SAVE_MODIFIED_ACTIONS_FILE_HEADER = "save_modified_actions_file"
MATCH_NUM_SAMPLES_HEADER = "match_num_samples"
SMOOTHING_ADDER_HEADER = "smoothing_adder"
NUM_EXPECTATION_SAMPLES_HEADER = "num_expectation_samples"
NUM_PERMUTATIONS_HEADER = "num_permutations"
BINARY_REWARDS_HEADER = "use_binary_rewards"
PRINT_ALL_STATS_FILE_HEADER = "print_all_stats"
NUM_STEP_SIZES_HEADER = "num_step_sizes"
SAMPLING_METHOD_HEADER = "sampling_type"
CALC_BANDITS_HEADER = "calculate_bandits"
MAKE_SUMMARY_PICKLES_HEADER = "make_summary_pickles"
MAKE_IPW_FILES_HEADER = "make_ipw_files"
RUN_PERMUTATION_TESTS_HEADER = "run_permutation_tests"
MAKE_SUMMARY_FILES_HEADER = "make_summary_files"

FORCE_ACTIONS_HEADER = "force_actions"

SMOOTHING_ADDER = 1;
MAX_WEIGHT = float("inf")

defaults_dictionary = {MAX_WEIGHT_HEADER: MAX_WEIGHT,
                       SAVE_MODIFIED_ACTIONS_FILE_HEADER: False,
                       MATCH_NUM_SAMPLES_HEADER: True,
                       SMOOTHING_ADDER_HEADER: SMOOTHING_ADDER,
                       NUM_EXPECTATION_SAMPLES_HEADER: 0,
                       NUM_PERMUTATIONS_HEADER: 5}

def get_json_arguments(configuration_file, add_defaults = True):
    config = {}
    if configuration_file.endswith("json"):
        with open(configuration_file) as json_file:
            config = json.load(json_file)
            
    if add_defaults:
        apply_defaults(config)
    return config

def apply_defaults(config):
    for param_key in defaults_dictionary:
        if param_key not in config:
            config[param_key] = defaults_dictionary[param_key]
            
            