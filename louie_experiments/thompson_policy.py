import sys
import csv
import random
import math
import numpy as np
import collections
from math import sqrt
from forced_actions import forced_actions
from enum import Enum
from bandit_data_format import *
from logistic_regression import *
from beta_bernoulli import *
from nig_normal import *
from output_format import *
from random_policy import *
from generate_single_bandit import *
import pandas as pd
import run_effect_size_simulations_beta

import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

class ActionSelectionMode(Enum):
    # Select action by probability it is best
    prob_is_best = 0

    # Select action in proportion to expected rewards
    expected_value = 1


def create_output_column_names_list(action_count):
    column_names = []

    column_names.extend([H_ALGO_ACTION,
        H_ALGO_OBSERVED_REWARD,
        H_ALGO_MATCH_OPTIMAL,
        H_ALGO_REGRET_EXPECTED,
        H_ALGO_REGRET_EXPECTED_CUMULATIVE])

    for idx in range(1, action_count+1):
        column_names.append(H_ALGO_ACTION_SUCCESS.format(idx))
        column_names.append(H_ALGO_ACTION_FAILURE.format(idx))
        column_names.append(H_ALGO_ESTIMATED_PROB.format(idx))

    for idx in range(1, action_count+1):
        column_names.append(H_ALGO_ACTION_SAMPLE.format(idx))
    column_names.append(H_ALGO_EXPLORING)
    return column_names


def create_headers(field_names, num_actions):
    # Construct output column header names
    field_names_out = field_names[:]
    field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                            H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                            H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])

    # not important, store the position to write high level header to output file
    group_header_parameters_index = len(field_names_out)

    for a in range(num_actions):
        field_names_out.append(H_ALGO_ACTION_SUCCESS.format(a + 1))
        field_names_out.append(H_ALGO_ACTION_FAILURE.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_PROB.format(a + 1))


    field_names_out.extend([H_ALGO_PROB_BEST_ACTION.format(a + 1) for a in range(num_actions)])
    field_names_out.append(H_ALGO_NUM_TRIALS)
    field_names_out.extend([H_ALGO_ACTION_SAMPLE.format(a + 1) for a in range(num_actions)])
    field_names_out.append(H_ALGO_CHOSEN_ACTION)

    # print group-level headers for readability
    group_header = ['' for i in range(len(field_names_out))]
    group_header[0] = "Input Data"
    group_header[len(field_names)] = "Algorithm's Performance"
    group_header[group_header_parameters_index] = "Model Parameters"

    return field_names_out, group_header


def write_performance(out_row, action, optimal_action, reward, sample_regret, cumulative_sample_regret, expected_regret,
                      cumulative_expected_regret):
    ''' write performance data (e.g. regret) '''
    out_row[H_ALGO_ACTION] = action + 1
    out_row[H_ALGO_OBSERVED_REWARD] = reward
    if isinstance(optimal_action, collections.Iterable):
        out_row[H_ALGO_MATCH_OPTIMAL] = 1 if (action + 1) in optimal_action else 0
    else:
        out_row[H_ALGO_MATCH_OPTIMAL] = 1 if optimal_action == (action + 1) else 0
    out_row[H_ALGO_SAMPLE_REGRET] = sample_regret
    out_row[H_ALGO_SAMPLE_REGRET_CUMULATIVE] = cumulative_sample_regret
    out_row[H_ALGO_REGRET_EXPECTED] = expected_regret
    out_row[H_ALGO_REGRET_EXPECTED_CUMULATIVE] = cumulative_expected_regret
    pass


def write_parameters(out_row, action, samples, models,
                     chosen_action_counts, num_actions,
                     num_trials_prob_best_action, context = None):
    ''' write parameters data (e.g. beta parameters)'''
    wroteParams = True
    for a in range(len(models)):
        try:
            if context != None:
                models[a].write_parameters(out_row, a, context)
            else:
                models[a].write_parameters(out_row, a)
        except:
            wroteParams = False
    
    if not wroteParams:
        try:
            index = 0
            for modelList in models:
                for model in modelList:
                    model.write_parameters(out_row, index, context)
                    index += 1
        except:
            pass
            

    # probability that each action is the best action
    # TODO: call a function to compute this value
    # for a in range(num_actions):
    #     out_row[H_ALGO_PROB_BEST_ACTION.format(a + 1)] = \
    #         float(chosen_action_counts[a]) / np.sum(chosen_action_counts)

    # number of repeated trials of Thompson Sampling to determine the
    # probability that each action is the best action
    out_row[H_ALGO_NUM_TRIALS] = num_trials_prob_best_action

    # samples for each action
    try:
        for a in range(num_actions):
            out_row[H_ALGO_ACTION_SAMPLE.format(a + 1)] = samples[a]
    except:
        index = 0
        for sampleList in samples:
            for sample in sampleList:
                out_row[H_ALGO_ACTION_SAMPLE.format(index + 1)] = sample
                index += 1

    # chosen action at this time step
    out_row[H_ALGO_CHOSEN_ACTION] = action + 1


def run_thompson_trial(context, num_samples, num_actions, models):
    '''
    Run Thompson Sampling many times using the specified Beta parameters.
    This is useful to compute several values in expectation, e.g. probability
    that each action is the best action, or the expected reward.
    :param context: Context features.
    :param num_samples: Number of times to run Thompson Sampling for.
    :param num_actions: Number of actions.
    :param models: The current model states for each action.
    '''
    ######################################################################################
    # NOTE: Uncomment these to print out model parameters for Thompson Sampling
    # NOTE: However, this will make performance quite a bit slower
    ######################################################################################
    ## tile the success and failure counts to generate beta samples
    ## efficiently using vectorized operations
    # samples = [models[a].draw_expected_value(context, num_samples) for a in range(num_actions)]

    ## generate a matrix of size (num_actions x num_trials) containing sampled expected values
    # samples = np.array(samples)

    ## take argmax of each row to get the chosen action
    # chosen_actions = np.argmax(samples, 0)

    # chosen_action_counts = np.zeros(num_actions)

    ## count how many times each action was chosen
    ## result is an array of size [num_actions] where value at
    ## i-th index is the number of times action index i was chosen.
    # bin_counts = np.bincount(chosen_actions)

    # chosen_action_counts[:len(bin_counts)] = bin_counts

    # return chosen_action_counts
    ######################################################################################
    return np.zeros(num_actions)

def calculate_thompson_single_bandit_factorial(source, num_actions_vector, dest, models,
                                               conditionsToActionIndex,
                                               epsilon = 0, get_context=get_context):
    '''
    Calculates non-contextual thompson sampling actions and weights. Assumes that we are making len(num_actions_vector)
    separate choices, and then combining them to get one final choice
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions_vector: vector where each entry is the number of actions for one of the individual bandits
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param conditionsToActionIndex: dictionary with keys that are tuples of conditions and values that is the larger action label for that set of condition choices
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
     :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)


    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, len(conditionsToActionIndex))

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []

        for row in reader:
            sample_number += 1

            # get context features
            context = get_context(row)

            should_update_posterior = True
            
            # first decide which arm we'd pull using Thompson
            # (do the random sampling, the max is the one we'd choose)
            all_condition_choices = []
            all_samples = []
            for experiment in range(len(num_actions_vector)):
                cur_num_actions = num_actions_vector[experiment]
                cur_models = models[experiment]
                samples = [cur_models[a].draw_expected_value(context) for a in range(cur_num_actions)]
                all_samples.append(samples)
                if epsilon > 0 and np.random.rand() < epsilon:
                    action = np.random.randint(cur_num_actions)
                else:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                all_condition_choices.append(action)
            action = conditionsToActionIndex[tuple(all_condition_choices)]
            

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(len(conditionsToActionIndex))]
            reward = observed_rewards[action]

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                for curModels, condition in zip(models,all_condition_choices):
                    curModels[condition].update_posterior(context, 2 * reward - 1)
                    



            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action_from_file = row[HEADER_OPTIMALACTION]
            if ';' in optimal_action_from_file:
                all_optimal_actions = [int(a) for a in optimal_action_from_file.split(';')]
            else:
                all_optimal_actions = [int(optimal_action_from_file)]
            optimal_action = all_optimal_actions[0] - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret
 
            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(len(conditionsToActionIndex))]
 
            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            #
            # # Run thompson sampling many times and calculate how much reward it would
            # # have gotten based on the chosen actions.
            chosen_action_counts = run_thompson_trial(context, num_trials_prob_best_action, len(conditionsToActionIndex), models)
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(len(conditionsToActionIndex)))
 
            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret
 
            write_performance(out_row, action, all_optimal_actions, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, all_samples, models,
                             chosen_action_counts, len(conditionsToActionIndex), 
                             num_trials_prob_best_action, context)

            writer.writerow(out_row)

        return chosen_actions, models

def calculate_thompson_single_bandit(source, num_actions, dest, models=None,
                                     action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                     relearn=True, epsilon = 0, get_context=get_context, batch_size = 1, burn_in_size = 1):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode. thompson_policy.ActionSelectionMode.prob_is_best = 0
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, num_actions)

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0
 
        initial_batch = True

        chosen_actions = []
        action_batch = []
        reward_batch = []

        for row in reader: #going through trials
            
            sample_number += 1
            #trial_number = num_steps_prev + sample_number

            # get context features
            context = get_context(row)
            if initial_batch:
                batch_size_curr = burn_in_size
            else:
                batch_size_curr = batch_size

            should_update_posterior = True

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                # first decide which arm we'd pull using Thompson
                # (do the random sampling, the max is the one we'd choose)
                samples = [models[a].draw_expected_value(context) for a in range(num_actions)]

                if epsilon > 0 and np.random.rand() < epsilon:
                    action = np.random.randint(num_actions)
                elif action_mode == ActionSelectionMode.prob_is_best:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                else:
                    # take action in proportion to expected rewards
                    # draw samples and normalize to use as a discrete distribution
                    # action is taken by sampling from this discrete distribution
                    probs = samples / np.sum(samples)
                    rand = np.random.rand()
                    for a in range(num_actions):
                        if rand <= probs[a]:
                            action = a
                            break
                        rand -= probs[a]

            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

                if relearn == False:
                    should_update_posterior = False

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            action_batch.append(action)
            reward_batch.append(reward)

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                if (sample_number % batch_size_curr == 0):
                   # print("updating batches")
                    #print("batch_size_curr, initial_batch, sample_number:", batch_size_curr, initial_batch, sample_number)
                    #print("action, reward len", len(action_batch), len(reward_batch))
                    for action, reward in zip(action_batch, reward_batch):#Update model based on a batch of actions and rewards
                        models[action].update_posterior(context, 2 * reward - 1)

                    action_batch = [] #reset batch
                    reward_batch = []
                    initial_batch = False

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)
                # save the model state in order so we can restore it
                # after switching to the true reward data.
                if sample_number == forced.time_step:
                    for a in range(num_actions):
                        models[a].save_state()

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action_from_file = row[HEADER_OPTIMALACTION]
            if ';' in optimal_action_from_file:
                all_optimal_actions = [int(a) for a in optimal_action_from_file.split(';')]
            else:
                all_optimal_actions = [int(optimal_action_from_file)]
            optimal_action = all_optimal_actions[0] - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret
 
            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]
 
            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            #
            # # Run thompson sampling many times and calculate how much reward it would
            # # have gotten based on the chosen actions.
            chosen_action_counts = run_thompson_trial(context, num_trials_prob_best_action, num_actions, models)
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(num_actions))
 
            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret
 
            write_performance(out_row, action, all_optimal_actions, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, samples, models,
                             chosen_action_counts, num_actions, num_trials_prob_best_action, context)

            writer.writerow(out_row)
        #print("sample_number", sample_number)
        return chosen_actions, models


def old_two_phase_random_thompson_policy(source, num_actions, dest,
                                    random_dur, models=None, random_start = 1,
                                    action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                    relearn=True, epsilon = 0, get_context=get_context, batch_size = 1, burn_in_size = 1):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param random_dur: the number of iterations that random policy will be used (must: random_dur+random_start-1 <= n)
    :param random_start: specifies the iteration in which the policy will switch to random policy (default=1)
    :param action_mode: Indicates how to select actions, see ActionSelectionMode. thompson_policy.ActionSelectionMode.prob_is_best = 0
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        count_n = len(list(reader))#sum(1 for _ in reader)
        inf.seek(0)
        reader = csv.DictReader(inf)

        if random_dur+random_start-1 > count_n:
            raise ValueError("random_dur+random_start-1 must be lower or equal to number of samples")

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, num_actions)

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0
 
        initial_batch = True

        chosen_actions = []
        action_batch = []
        reward_batch = []

        for row in reader: #going through trials
            sample_number += 1
            #trial_number = num_steps_prev + sample_number

            # get context features
            context = get_context(row)
            if initial_batch:
                batch_size_curr = burn_in_size
            else:
                batch_size_curr = batch_size

            should_update_posterior = True
            if sample_number >= random_start and sample_number < (random_dur+random_start):
                action = np.random.randint(num_actions)
                samples = [models[a].draw_expected_value() for a in range(num_actions)]
            elif len(forced.actions) == 0 or sample_number > len(forced.actions):
                # first decide which arm we'd pull using Thompson
                # (do the random sampling, the max is the one we'd choose)
                samples = [models[a].draw_expected_value() for a in range(num_actions)]

                if epsilon > 0 and np.random.rand() < epsilon:
                    action = np.random.randint(num_actions)
                elif action_mode == ActionSelectionMode.prob_is_best:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                else:
                    # take action in proportion to expected rewards
                    # draw samples and normalize to use as a discrete distribution
                    # action is taken by sampling from this discrete distribution
                    probs = samples / np.sum(samples)
                    rand = np.random.rand()
                    for a in range(num_actions):
                        if rand <= probs[a]:
                            action = a
                            break
                        rand -= probs[a]

            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

                if relearn == False:
                    should_update_posterior = False

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            action_batch.append(action)
            reward_batch.append(reward)

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                if (sample_number % batch_size_curr == 0):
                   # print("updating batches")
                    #print("batch_size_curr, initial_batch, sample_number:", batch_size_curr, initial_batch, sample_number)
                    #print("action, reward len", len(action_batch), len(reward_batch))
                    for action, reward in zip(action_batch, reward_batch):#Update model based on a batch of actions and rewards
                        models[action].update_posterior(2 * reward - 1)

                    action_batch = [] #reset batch
                    reward_batch = []
                    initial_batch = False

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)
                # save the model state in order so we can restore it
                # after switching to the true reward data.
                if sample_number == forced.time_step:
                    for a in range(num_actions):
                        models[a].save_state()

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            #logging.info(out_row)
            #logging.info(len(reader.fieldnames))
            ''' write performance data (e.g. regret) '''
            optimal_action_from_file = row[HEADER_OPTIMALACTION]
            if ';' in optimal_action_from_file:
                all_optimal_actions = [int(a) for a in optimal_action_from_file.split(';')]
            else:
                all_optimal_actions = [int(optimal_action_from_file)]
            optimal_action = all_optimal_actions[0] - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]

            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] #* num_trials_prob_best_action
            expected_regret = true_probs[action] - optimal_expected_reward
            cumulative_expected_regret += expected_regret
            chosen_action_counts = 0
 
            write_performance(out_row, action, all_optimal_actions, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, samples, models,
                             chosen_action_counts, num_actions, num_trials_prob_best_action, context)

            writer.writerow(out_row)
        #print("sample_number", sample_number)
        return chosen_actions, chosen_actions,models


def create_output_list(selected_action, optimal_action, reward, expected_regret,
        cumulative_expected_regret, models, samples, is_exploring, distribution='bernoulli'):

    if distribution != 'bernoulli':
        raise ValueError('Not implemented yet!')

    data_list = []
    column_names = []

    data_list.append(selected_action+1)
    data_list.append(reward)
    if isinstance(optimal_action, collections.abc.Iterable):
        data_list.append(1 if (selected_action + 1) in optimal_action else 0)
    else:
        data_list.append(1 if optimal_action == (selected_action + 1) else 0)
    data_list.append(expected_regret)
    data_list.append(cumulative_expected_regret)
    for model in models:
        ls = model.get_parameters()
        data_list.append(ls[0]) #number of success for a model
        data_list.append(ls[1]) #number of failures for a model
        data_list.append(ls[2]) #estimated reward probability
    for sample in samples:
        data_list.append(sample)

    data_list.append(is_exploring)
    return data_list


def two_phase_random_thompson_policy(prob_per_arm, users_count,
                                    random_dur, models=None, random_start = 0,
                                    action_mode=ActionSelectionMode.prob_is_best,
                                    forced=forced_actions(), relearn=True,
                                    epsilon = 0, get_context=get_context,
                                    batch_size = 1, decreasing_epsilon=0):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param random_dur: the number of iterations that random policy will be used (must: random_dur+random_start-1 <= n)
    :param random_start: specifies the iteration in which the policy will switch to random policy (default=1)
    :param action_mode: Indicates how to select actions, see ActionSelectionMode. thompson_policy.ActionSelectionMode.prob_is_best = 0
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time
    :param decreasing_epsilon: Optional, if epsilon>0 and decreasing_epsilon=1, epsilon will decrease with 1/sqrt(n) rate
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed

    is_exploring = None
    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    num_actions = len(prob_per_arm)
    if random_dur+random_start > users_count:
            raise ValueError("random_dur+random_start must be lower or equal to number of samples")

    optimal_action = ";".join(str(x) for x in np.nonzero([False]+(prob_per_arm == np.max(prob_per_arm)).tolist())[0].tolist())

    sample_number = 0
    cumulative_expected_regret = 0

    chosen_actions = []
    action_batch = []
    reward_batch = []
    simulated_results = []

    for row in range(users_count): #going through trials
        sample_number += 1

        batch_size_curr = batch_size
        should_update_posterior = True

        if sample_number > random_start and sample_number <= (random_dur+random_start):
            action = np.random.randint(num_actions)
            samples = [models[a].draw_expected_value() for a in range(num_actions)]
        elif len(forced.actions) == 0 or sample_number > len(forced.actions):
            # first decide which arm we'd pull using Thompson
            # (do the random sampling, the max is the one we'd choose)
            samples = [models[a].draw_expected_value() for a in range(num_actions)]

            if decreasing_epsilon:
                epslion_new = epsilon / sqrt(sample_number-len(forced.actions))
            else:
                epslion_new = epsilon

            if epslion_new > 0 and np.random.rand() < epslion_new:
                is_exploring = 1
                action = np.random.randint(num_actions)
            elif action_mode == ActionSelectionMode.prob_is_best:
                # find the max of samples[i] etc and choose an arm
                is_exploring = 0
                action = np.argmax(samples)
            else:
                # take action in proportion to expected rewards
                # draw samples and normalize to use as a discrete distribution
                # action is taken by sampling from this discrete distribution
                probs = samples / np.sum(samples)
                rand = np.random.rand()
                for a in range(num_actions):
                    if rand <= probs[a]:
                        action = a
                        break
                    rand -= probs[a]
        else:
            samples = [0 for a in range(num_actions)]
            # take forced action if requested
            action = forced.actions[sample_number - 1]

            if relearn == False:
                should_update_posterior = False

        sample = samples[action]
        reward = models[action].perform_bernoulli_trials(prob_per_arm[action])

        action_batch.append(action)
        reward_batch.append(reward)

        if should_update_posterior:
            # update posterior distribution with observed reward
            # converted to range {-1,1}
            if ((sample_number-(random_dur+random_start)) % batch_size_curr == 0) and (sample_number < random_start or sample_number >= (random_dur+random_start)):
                for action, reward in zip(action_batch, reward_batch):#Update model based on a batch of actions and rewards
                    models[action].update_posterior(2 * reward - 1)

                action_batch = [] #reset batch
                reward_batch = []

        # only return action chosen up to specified time step
        if forced.time_step > 0 and sample_number <= forced.time_step:
            chosen_actions.append(action)
            # save the model state in order so we can restore it
            # after switching to the true reward data.
            if sample_number == forced.time_step:
                for a in range(num_actions):
                    models[a].save_state()

        if ';' in optimal_action:
            all_optimal_actions = [int(a) for a in optimal_action.split(';')]
        else:
            all_optimal_actions = [int(optimal_action)]

        # # The oracle always chooses the best arm, thus expected reward
        # # is simply the probability of that arm getting a reward.
        optimal_expected_reward = prob_per_arm[all_optimal_actions[0]-1] #* num_trials_prob_best_action
        expected_regret = prob_per_arm[action] - optimal_expected_reward
        cumulative_expected_regret += expected_regret
        chosen_action_counts = 0

        measurements = create_output_list(action, all_optimal_actions,
            reward, expected_regret, cumulative_expected_regret, models, samples, is_exploring)

        simulated_results.append(measurements)

    column_names = create_output_column_names_list(num_actions)
    return simulated_results, column_names, models


def ppd_two_phase_random_thompson_policy(prob_per_arm, users_count,
                                    random_dur, models=None, random_start = 0,
                                    action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                    relearn=True, c = 0.1, get_context=get_context, batch_size = 1, resample = True):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param random_dur: the number of iterations that random policy will be used (must: random_dur+random_start-1 <= n)
    :param random_start: specifies the iteration in which the policy will switch to random policy (default=1)
    :param action_mode: Indicates how to select actions, see ActionSelectionMode. thompson_policy.ActionSelectionMode.prob_is_best = 0
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param c: c parameter for ppd 
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed

    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    num_actions = len(prob_per_arm)
    if random_dur+random_start > users_count:
            raise ValueError("random_dur+random_start must be lower or equal to number of samples")

    optimal_action = ";".join(str(x) for x in np.nonzero([False]+(prob_per_arm == np.max(prob_per_arm)).tolist())[0].tolist())

    sample_number = 0
    cumulative_expected_regret = 0
    is_exploring = None

    chosen_actions = []
    action_batch = []
    reward_batch = []
    simulated_results = []

    for row in range(users_count): #going through trials
        sample_number += 1

        batch_size_curr = batch_size

        should_update_posterior = True
        if sample_number > random_start and sample_number <= (random_dur+random_start):
            action = np.random.randint(num_actions)
            samples = [models[a].draw_expected_value() for a in range(num_actions)]
        elif len(forced.actions) == 0 or sample_number > len(forced.actions):
            # first decide which arm we'd pull using Thompson
            # (do the random sampling, the max is the one we'd choose)
            samples = [models[a].draw_expected_value() for a in range(num_actions)]

            diff = np.abs(samples[0] -samples[1])

            if diff < c:
        #        print("exploring, diff, thresh", diff, thresh)
                    # take a random action
                is_exploring = 1
                action = np.random.randint(num_actions)
            elif action_mode == ActionSelectionMode.prob_is_best:
                # find the max of samples[i] etc and choose an arm
                is_exploring = 0
                if resample == True:
                    samples = [models[a].draw_expected_value() for a in range(num_actions)]
                action = np.argmax(samples)
            else:
                # take action in proportion to expected rewards
                # draw samples and normalize to use as a discrete distribution
                # action is taken by sampling from this discrete distribution
                probs = samples / np.sum(samples)
                rand = np.random.rand()
                for a in range(num_actions):
                    if rand <= probs[a]:
                        action = a
                        break
                    rand -= probs[a]
        else:
            samples = [0 for a in range(num_actions)]
            # take forced action if requested
            action = forced.actions[sample_number - 1]

            if relearn == False:
                should_update_posterior = False

        sample = samples[action]
        reward = models[action].perform_bernoulli_trials(prob_per_arm[action])

        action_batch.append(action)
        reward_batch.append(reward)

        if should_update_posterior:
            # update posterior distribution with observed reward
            # converted to range {-1,1}
            if ((sample_number-(random_dur+random_start)) % batch_size_curr == 0) and (sample_number < random_start or sample_number >= (random_dur+random_start)):
                for action, reward in zip(action_batch, reward_batch):#Update model based on a batch of actions and rewards
                    models[action].update_posterior(2 * reward - 1)

                action_batch = [] #reset batch
                reward_batch = []

        # only return action chosen up to specified time step
        if forced.time_step > 0 and sample_number <= forced.time_step:
            chosen_actions.append(action)
            # save the model state in order so we can restore it
            # after switching to the true reward data.
            if sample_number == forced.time_step:
                for a in range(num_actions):
                    models[a].save_state()

        if ';' in optimal_action:
            all_optimal_actions = [int(a) for a in optimal_action.split(';')]
        else:
            all_optimal_actions = [int(optimal_action)]

        # # The oracle always chooses the best arm, thus expected reward
        # # is simply the probability of that arm getting a reward.
        optimal_expected_reward = prob_per_arm[all_optimal_actions[0]-1] #* num_trials_prob_best_action
        expected_regret = prob_per_arm[action] - optimal_expected_reward
        cumulative_expected_regret += expected_regret
        chosen_action_counts = 0

        measurements = create_output_list(action, all_optimal_actions,
            reward, expected_regret, cumulative_expected_regret, models, samples, is_exploring)

        simulated_results.append(measurements)

    column_names = create_output_column_names_list(num_actions)
    return simulated_results, column_names, models


def calculate_thompson_switch_to_fixed_policy(source, num_actions, dest, num_actions_before_switch, models=None, 
                                     switch_to_best_if_nonsignificant=True,
                                     action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                     epsilon = 0):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)
    arm_to_choose = None
    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]
    out_rows = []
    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, num_actions)

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []

        for row in reader:
            sample_number += 1

            # get context features
            context = get_context(row)

            should_update_posterior = True
            if arm_to_choose != None:
                samples = [0 for a in range(num_actions)]
                # take forced action
                action = arm_to_choose
            elif len(forced.actions) == 0 or sample_number > len(forced.actions):
                # first decide which arm we'd pull using Thompson
                # (do the random sampling, the max is the one we'd choose)
                samples = [models[a].draw_expected_value(context) for a in range(num_actions)]

                if epsilon > 0 and np.random.rand() < epsilon:
                    action = np.random.randint(num_actions)
                elif action_mode == ActionSelectionMode.prob_is_best:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                else:
                    # take action in proportion to expected rewards
                    # draw samples and normalize to use as a discrete distribution
                    # action is taken by sampling from this discrete distribution
                    probs = samples / np.sum(samples)
                    rand = np.random.rand()
                    for a in range(num_actions):
                        if rand <= probs[a]:
                            action = a
                            break
                        rand -= probs[a]

            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                models[action].update_posterior(context, 2 * reward - 1)

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)
                # save the model state in order so we can restore it
                # after switching to the true reward data.
                if sample_number == forced.time_step:
                    for a in range(num_actions):
                        models[a].save_state()

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action_from_file = row[HEADER_OPTIMALACTION]
            if ';' in optimal_action_from_file:
                all_optimal_actions = [int(a) for a in optimal_action_from_file.split(';')]
            else:
                all_optimal_actions = [int(optimal_action_from_file)]
            optimal_action = all_optimal_actions[0] - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret
 
            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]
 
            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            #
            # # Run thompson sampling many times and calculate how much reward it would
            # # have gotten based on the chosen actions.
            chosen_action_counts = run_thompson_trial(context, num_trials_prob_best_action, num_actions, models)
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(num_actions))
 
            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret
 
            write_performance(out_row, action, all_optimal_actions, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, samples, models,
                             chosen_action_counts, num_actions, num_trials_prob_best_action, context)

            writer.writerow(out_row)
            out_rows.append(out_row)

            if sample_number == num_actions_before_switch:
                # Time to decide what to switch to - need to run a significance test
                output_file = pd.DataFrame.from_records(out_rows)
                cur_row = run_effect_size_simulations_beta.make_stats_row_from_df(output_file, False)
                if cur_row['pvalue'] < .05 or switch_to_best_if_nonsignificant:
                    #switch to best 
                    arm_to_choose = np.argmax([cur_row['mean_1'],cur_row['mean_2']])
                else:
                    # otherwise, we choose one of the arms to pull the rest of the time uniformly at random
                    # (as if we decided to stick with whatever business as usual was, and we're averaging
                    #  across possible business as usual possibilities)
                    arm_to_choose = np.random.randint(num_actions)

        return chosen_actions, models

def estimate_probability_condition_assignment(context, num_samples, num_actions, models):
    if num_samples > 0:
        samples = [models[a].draw_expected_value(context, num_samples) for a in range(num_actions)]
    
        ## generate a matrix of size (num_actions x num_trials) containing sampled expected values
        samples = np.array(samples)
    
        ## take argmax of each row to get the chosen action
        chosen_actions = np.argmax(samples, 0)
    
        ## count how many times each action was chosen
        ## result is an array of size [num_actions] where value at
        ## i-th index is the number of times action index i was chosen.
        chosen_action_counts = np.bincount(chosen_actions, minlength=num_actions)
    else:
        chosen_action_counts = [0]*num_actions
    return chosen_action_counts

def calculate_thompson_single_bandit_empirical_params(arm_1_rewards, arm_2_rewards, num_actions, dest, models=None,
                                     action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                     relearn=True, epsilon = 0):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param arm_1_rewards: actual rewards for 1st arm, as a list 
    :param arm_2_rewards: actual rewards for 2nd arm, as a list 
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    :param epsilon: Optional, if > 0 then we choose a random action epsilon proportion of the time. [if epsilon >= 1 then always random]
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    with open(dest, 'w', newline='') as outf:
        
        # Construct output column header names
        field_names = []
        field_names_out, group_header = create_headers(field_names, num_actions)

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []

        while True:
            sample_number += 1

            # get context features
            context = None

            should_update_posterior = True

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                # first decide which arm we'd pull using Thompson
                # (do the random sampling, the max is the one we'd choose)
                samples = [models[a].draw_expected_value(context) for a in range(num_actions)]

                if epsilon > 0 and np.random.rand() < epsilon: #proability clipping -------> allows also for completely uniform drawing when epsilon >= 1
                    action = np.random.randint(num_actions)
                elif action_mode == ActionSelectionMode.prob_is_best:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                else:
                    # take action in proportion to expected rewards
                    # draw samples and normalize to use as a discrete distribution
                    # action is taken by sampling from this discrete distribution
                    probs = samples / np.sum(samples)
                    rand = np.random.rand()
                    for a in range(num_actions):
                        if rand <= probs[a]:
                            action = a
                            break
                        rand -= probs[a]

            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

                if relearn == False:
                    should_update_posterior = False

            # get reward signals
            if action == 0:
                queue = arm_1_rewards
            else:
                queue = arm_2_rewards
                
            if len(queue) == 0:
                break; # Can't sample anymore
            else:
                reward = queue.pop(0)

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                models[action].update_posterior(context, 2 * reward - 1)

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)
                # save the model state in order so we can restore it
                # after switching to the true reward data.
                if sample_number == forced.time_step:
                    for a in range(num_actions):
                        models[a].save_state()

            # copy the input data to output file
            out_row = {}


            ''' write performance data (e.g. regret) '''
            optimal_action = -1
            optimal_action_reward = -1
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            true_probs = [0 for a in range(num_actions)]

            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            #
            # # Run thompson sampling many times and calculate how much reward it would
            # # have gotten based on the chosen actions.
            chosen_action_counts = run_thompson_trial(context, num_trials_prob_best_action, num_actions, models)
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(num_actions))

            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret

            write_performance(out_row, action, optimal_action, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, samples, models,
                             chosen_action_counts, num_actions, num_trials_prob_best_action)

            writer.writerow(out_row)

        return chosen_actions, models


def switch_bandit_thompson(immediate_input, true_input, immediate_output,
                           true_output, time_step, action_mode, relearn=True,
                           use_regression=False, num_actions=3, Lambda=1):
    '''
    Run the algorithm on immediate-reward input up to specified time step then switch to the true-reward input and
    recompute policy by keeping the previously taken actions and matching with true rewards instead.
    :param immediate_input: The immediate-reward input file.
    :param true_input: The true-reward input file.
    :param immediate_output: The result output file from applying the algorithm to the immediate input.
    :param true_output: The result output file from applying the algorithm to the true input.
    :param time_step: The time step to switch bandit.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
    :param relearn: At switch time, whether the algorithm will relearn from beginning.
    :param use_regression: Optional, indicate whether to use logistic regression to model reward distribution.
    :param num_actions: The number of actions in this bandit.
    :param Lambda: The prior inverse variance of the regression weights if regression is used.
    '''

    if use_regression:
        models = [RLogReg(D=NUM_FEATURES, Lambda=Lambda) for _ in range(num_actions)]
    else:
        models = [BetaBern(success=1, failure=1) for _ in range(num_actions)]

    # Run for 20 time steps on the immediate reward input
    chosen_actions, models = calculate_thompson_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        models,
        action_mode=action_mode,
        forced=forced_actions(time_step))

    # reset model state so that the algorithm forgets what happens
    for a in range(num_actions):
        models[a].reset_state()

    # Switch to true reward input, forcing actions taken previously
    calculate_thompson_single_bandit(
        true_input,
        num_actions,
        true_output,
        models,
        action_mode,
        forced_actions(actions=chosen_actions),
        relearn=relearn)


def switch_bandit_random_thompson(immediate_input, true_input, immediate_output,
                                  true_output, time_step, action_mode,
                                  relearn=True, use_regression=False,
                                  num_actions=3, Lambda=1):
    '''
    Similar to switch_bandit_thompson except that Random policy is run on the immediate data
    instead and thompson takes over once the switch happens.
    :param relearn: At switch time, whether the algorithm will relearn from beginning.
    '''

    if use_regression:
        models = [RLogReg(D=NUM_FEATURES, Lambda=Lambda) for _ in range(num_actions)]
    else:
        models = [BetaBern(success=1, failure=1) for _ in range(num_actions)]

    chosen_actions = calculate_random_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        forced=forced_actions(time_step))

    # Switch to true reward input, forcing actions taken previously
    calculate_thompson_single_bandit(
        true_input,
        num_actions,
        true_output,
        models,
        action_mode,
        forced_actions(actions=chosen_actions),
        relearn=relearn)


def main():
    num_actions = 3

    # init with inverse variance
    models = [RLogReg(D = NUM_FEATURES, Lambda = 1) for cond in range(num_actions)]

    #calculate_thompson_single_bandit('simulated_single_bandit_input.csv', 3, 'simulated_single_bandit_thompson.csv')
    calculate_thompson_single_bandit('contextual_single_bandit.csv', 3, 'contextual_single_bandit_thompson.csv', models)

if __name__ == "__main__":
    main()

