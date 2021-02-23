'''
Created on Aug 25, 2017

@author: rafferty


Contains functions for changing the order of the samples in a rewards file based
on particular criteria. Useful for investigating questions about a slightly different part
of the population arriving at one time versus another.
'''
import csv
import numpy as np
import math

PREFER_WORSE = False # When true, means that rewarding tends to have worse rewards coming first rather than better

def reorder_rewards_by_quartile(reward_in_file, reward_out_file, ordering_function,softmax_beta):
    with open(reward_in_file, newline='') as inf, open(reward_out_file, 'w', newline='') as outf:
        reader = csv.DictReader(inf)
        field_names = reader.fieldnames
        
        # We'll need to read the whole file into memory to be able to access random lines when
        # writing
        samples = []
        values_for_ordering = []
        for row in reader:
            samples.append(row)
            value_for_ordering = ordering_function(row)
            values_for_ordering.append(value_for_ordering)
        # then, need to find out a sorted order
        index_array = np.argsort(values_for_ordering)
        # could swap this out for a different funtion
        probabilities = weight_by_quartile(index_array, softmax_beta)
        permutation = np.random.choice(samples, len(samples), replace=False, p=probabilities)
        
        writer = csv.DictWriter(outf, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(permutation)

def weight_by_quartile(index_array, softmax_beta, num_diff_weights = 4):
    quartile_start_points = [int(round(len(index_array)*i/num_diff_weights)) for i in range(num_diff_weights)]
    quartile_weights = np.exp(np.arange(num_diff_weights)*softmax_beta)
    quartile_prob = quartile_weights / sum(quartile_weights)
    if PREFER_WORSE:
        quartile_prob = np.flip(quartile_prob, 0)
        print("Flipped quartile prob: ", quartile_prob)
    probabilities = np.zeros(len(index_array))
    for i in range(len(index_array)):
        quartile_index = get_quartile_index(i, len(index_array), quartile_start_points)
        cur_prob = quartile_prob[quartile_index] / get_num_items_in_quartile(quartile_index, len(index_array), quartile_start_points) 
        probabilities[index_array[i]] = cur_prob
    return probabilities
        
        
def get_num_items_in_quartile(quartile_index, num_items, quartile_start_points):
    if quartile_index + 1 == len(quartile_start_points):
        return num_items - quartile_start_points[-1]
    else:
        return quartile_start_points[quartile_index + 1] - quartile_start_points[quartile_index]
    
def get_quartile_index(rank, num_items, quartile_start_points):
    for i in range(1, len(quartile_start_points)):
        if rank < quartile_start_points[i]:
            return i - 1
    return len(quartile_start_points) - 1
   
def order_by_named_column(column_name):
    return lambda row: float(row[column_name])

def order_by_sum_of_rewards():
    return lambda row: float(row['Action1OracleActualReward']) + float(row['Action2OracleActualReward'])

def get_reordering_fn(reordering_fn_specifier):
    '''Takes in a string (command line parameter) that describes how to reorder rewards. 
    Sets any global variables for enforcing that reordering, and returns the function that should be
    used for the reordering. Returns None if specifier doesn't match any known specifier.'''
    global PREFER_WORSE
    reordering_fn = None
    if reordering_fn_specifier == "PreferBetterAction1":
        reordering_fn = order_by_named_column('Action1OracleActualReward')
    elif reordering_fn_specifier == "PreferBetterAction2":
        reordering_fn = order_by_named_column('Action2OracleActualReward')
    elif reordering_fn_specifier == "PreferBetterActions":
        reordering_fn = order_by_sum_of_rewards()
    elif reordering_fn_specifier == "PreferWorseAction1":
        PREFER_WORSE = True
        reordering_fn = order_by_named_column('Action1OracleActualReward')
    elif reordering_fn_specifier == "PreferWorseAction2":
        PREFER_WORSE = True
        reordering_fn =order_by_named_column('Action2OracleActualReward')
    elif reordering_fn_specifier == "PreferWorseActions":
        PREFER_WORSE = True
        reordering_fn = order_by_sum_of_rewards()
    return reordering_fn

def main():
    '''Debugging only'''
    reward_in_file = '/Users/rafferty/banditalgorithms/data/effectSizeDebugging/tbb_rewards_393_1.csv'
    reward_out_file = '/Users/rafferty/banditalgorithms/data/effectSizeDebugging/reordered_tbb_rewards_393_1.csv'
    ordering_function = get_reordering_fn("PreferWorseActions")
    reorder_rewards_by_quartile(reward_in_file, reward_out_file, ordering_function,0.5)

if __name__ == "__main__":
    main()