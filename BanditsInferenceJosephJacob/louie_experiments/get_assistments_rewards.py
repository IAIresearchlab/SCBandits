'''
Created on Nov 3, 2017

@author: rafferty
'''
import csv
import numpy as np

#Modification of the code according to other conditions

EXPERIMENT_ID_HEADER = 'Eligible';
CONDITION_HEADER = 'Arm'
#EXPERIENCED_CONDITION_HEADER = 'ExperiencedCondition'
#VIDEO_HEADER = 'Could See Video'

def read_assistments_rewards(data_file, reward_header, experiment_identifier, is_cost = False):  
    arm_1_identifier = '0'
    arm_1_rewards = []
    arm_2_rewards = []
    with open(data_file) as inf: #TestData a fake db created with Jacob
        reader = csv.DictReader(inf, delimiter=';') #pay attention to delimiter & UTF code
        for row in reader:
            print(row)
            if row[EXPERIMENT_ID_HEADER].strip() == experiment_identifier.strip():
                # Check if we have a value for this student - may not have values for all students

                if row[reward_header].strip() != "":
                    # Include this row if the student experienced a condition
#                    if row[EXPERIENCED_CONDITION_HEADER] == 'TRUE' and \
#                    not (row[VIDEO_HEADER] == '0' and row[CONDITION_HEADER] == 'E'):
                    queue = arm_1_rewards
                    if arm_1_identifier != row[CONDITION_HEADER]:
                        queue = arm_2_rewards
                    cur_reward = float(row[reward_header])
                    if is_cost:
                        cur_reward *= -1
                    queue.append(cur_reward)
    
        
    if sum(arm_1_rewards)/len(arm_1_rewards) > sum(arm_2_rewards)/len(arm_2_rewards):
        return arm_1_rewards, arm_2_rewards
    else:
        return arm_2_rewards, arm_1_rewards

def read_pcrs_rewards(data_file, reward_header, condition_header, is_cost = False):  
    arm_1_identifier = '0'
    arm_1_rewards = []
    arm_2_rewards = []
    with open(data_file) as inf: 
        reader = csv.DictReader(inf) #pay attention to delimiter & UTF code

        for row in reader:
            #print(row)
            #print(row.keys()[0])
            #if row[EXPERIMENT_ID_HEADER].strip() == experiment_identifier.strip():
                # Check if we have a value for this student - may not have values for all students

            if row[reward_header].strip() != "" and row[reward_header].strip() != "NA":
                queue = arm_1_rewards
                if arm_1_identifier != row[condition_header]:
                    queue = arm_2_rewards
                cur_reward = float(row[reward_header])
                if is_cost:
                    cur_reward *= -1
                queue.append(cur_reward)
    
        
    if np.nansum(arm_1_rewards)/len(arm_1_rewards) > np.nansum(arm_2_rewards)/len(arm_2_rewards):
        return arm_1_rewards, arm_2_rewards
    else:
        return arm_2_rewards, arm_1_rewards


                   
if __name__ == "__main__":
    # Debugging only
    #arm_1_rewards, arm_2_rewards = read_assistments_rewards("../../../empirical_data/TestData.csv", 'Outcome1', '255116')
    reward_file = "../../../empirical_data/experiments_data.csv"
    reward_header = "Y2"
    condition_header = 'motivate_final'
    arm_1_rewards, arm_2_rewards = read_pcrs_rewards(reward_file, reward_header, condition_header) #Load emprical rewards from rewards_file 
    print(len(arm_1_rewards))
    print(len(arm_2_rewards))
    print(sum(arm_1_rewards)/len(arm_1_rewards))
    print(sum(arm_2_rewards)/len(arm_2_rewards))