import matplotlib
matplotlib.use('Agg')
import pickle
import os
#import ipdb
import statsmodels.stats.power as smp
from rectify_vars_and_wald_functions import *
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(1, '../../../louie_experiments/')
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
import read_config
from output_format import H_ALGO_ACTION_FAILURE, H_ALGO_ACTION_SUCCESS, H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD
from output_format import H_ALGO_ESTIMATED_MU, H_ALGO_ESTIMATED_V, H_ALGO_ESTIMATED_ALPHA, H_ALGO_ESTIMATED_BETA
from output_format import H_ALGO_PROB_BEST_ACTION, H_ALGO_NUM_TRIALS
import beta_bernoulli
#import thompson_policy
from pathlib import Path

EPSILON_PROB = .000001

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def hist_pval(df = None, to_check = None, to_check_unif = None, to_check_ts = None, n = None, num_sims = None, load_df = True, \
                     title = None, plot = True):
    '''
    TODO rename to_check_ipw to to_check_ipw_wald_stat
    '''
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        with open(to_check_ts, 'rb') as f:
            df_ts = pickle.load(f)

    #print(data)
    
    step_sizes = df['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    if plot == True:
        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(14.5, 10.5)
        ax = ax.ravel()
    i = 0
    percenticle_dict_left = {}
    percentile_dict_right = {}
    for num_steps in step_sizes:

        df_unif_for_num_steps = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps = df[df['num_steps'] == num_steps]
        df_ts_for_num_steps = df_ts[df_ts['num_steps'] == num_steps]

        mle_mean1 = np.mean(df_for_num_steps['mean_1'])
        mle_mean2 = np.mean(df_for_num_steps['mean_2'])
        unif_mean1 = np.mean(df_unif_for_num_steps['mean_1'])
        unif_mean2 = np.mean(df_unif_for_num_steps['mean_2'])
        
        
        df_for_num_steps_pval = df_for_num_steps['pvalue']
        df_unif_for_num_steps_pval = df_unif_for_num_steps['pvalue']
        df_ts_for_num_steps_pval = df_ts_for_num_steps['pvalue']
      #  df_unif_for_num_steps = np.ma.masked_invalid(df_unif_for_num_steps)
        #print(np.mean(df_unif_for_num_steps))
        if plot == True:
            #ax[i].hist(df_unif_for_num_steps, density = True)
            ax[i].hist(df_unif_for_num_steps_pval, normed = False, alpha = 0.5, \
              label = "Uniform")

            ax[i].hist(df_for_num_steps_pval, \
              normed = False, alpha = 0.5, \
              label = "Epsilon Greedy")

            ax[i].hist(df_ts_for_num_steps_pval, \
              normed = False, alpha = 0.5, \
              label = "Thompson Sampling")
            
            ax[i].set_xlabel("Pvalue for number of participants = {} = {}".format(size_vars[i], num_steps))
 
       #     mu = 0
        #    variance = 1
         #   sigma = np.sqrt(variance)
          #  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
           # ax[i].plot(x, stats.norm.pdf(x, mu, sigma))
            ax[i].legend()
            

        i+=1    
    if plot == True:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
       # if not os.path.isdir("plots"):
        #    os.path.mkdir("plots")
        print("saving to ", "pval_hist/{}.png".format(title))
        fig.savefig("pval_hist/{}.png".format(title))

        #plt.show()
        plt.clf()
        plt.close()
    

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

def plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon, n):
        fig_h, ax_h = plt.subplots()
        proportions_unif = df_for_num_steps_unif['sample_size_1'] / num_steps
        proportions_eg0pt1 = df_for_num_steps_eg0pt1['sample_size_1'] / num_steps
        proportions_eg0pt3 = df_for_num_steps_eg0pt3['sample_size_1'] / num_steps
        proportions_ts = df_for_num_steps_ts['sample_size_1'] / num_steps
        
        ax_h.hist(proportions_eg0pt1, alpha = 0.5, label = "Epsilon Greedy 0.1")
        ax_h.hist(proportions_eg0pt3, alpha = 0.5, label = "Epsilon Greedy 0.3")
        ax_h.hist(proportions_unif, alpha = 0.5, label = "Uniform Random")
        ax_h.hist(proportions_ts, alpha = 0.5, label = "Thompson Sampling")
        ax_h.legend()
        fig_h.suptitle("Histogram of Proportion of {} Participants Assigned to Condition 1 Across 500 Simulations".format(num_steps))
       # rows = ["Areferg"]
       # columns = ["Berger"]
       # cell_text = ["ergerg"]
       # the_table = ax_h.table(cellText=cell_text,
         #             rowLabels=rows,
        #              colLabels=columns,
          #            loc='right')

      #  fig_h.subplots_adjust(left=0.2, wspace=0.4)
        data = np.random.uniform(0, 1, 80).reshape(20, 4)
        mean_ts = np.mean(proportions_ts)
        var_ts = np.var(proportions_ts)

        mean_eg0pt1 = np.mean(proportions_eg0pt1)
        mean_eg0pt3 = np.mean(proportions_eg0pt3)
        var_eg0pt1 = np.var(proportions_eg0pt1)
        var_eg0pt3 = np.var(proportions_eg0pt3)

        prop_lt_25_eg0pt1 = np.sum(proportions_eg0pt1 < 0.25) / len(proportions_eg0pt1)
        prop_lt_25_eg0pt3 = np.sum(proportions_eg0pt3 < 0.25) / len(proportions_eg0pt3)
        prop_lt_25_ts = np.sum(proportions_ts < 0.25) / len(proportions_ts)

       # prop_gt_25_lt_5_eg = np.sum(> proportions > 0.25) / len(proportions)
       # prop_gt_25_lt_5_ts = np.sum(> proportions_ts > 0.25) / len(proportions_ts)

        data = [[mean_ts, var_ts, prop_lt_25_ts],\
         [mean_eg0pt1, var_eg0pt1, prop_lt_25_eg0pt1],\
         [mean_eg0pt3, var_eg0pt3, prop_lt_25_eg0pt3]]


        final_data = [['%.3f' % j for j in i] for i in data] #<0.25, 0.25< & <0.5, <0.5 & <0.75, <0.75 & <1.0                                                                                                                 
        #table.auto_set_font_size(False)
      #  table.set_fontsize(7)
      #  table.auto_set_column_width((-1, 0, 1, 2, 3))
        table = ax_h.table(cellText=final_data, colLabels=['Mean', 'Variance', 'prop < 0.25'], rowLabels = ["Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"], loc='bottom', cellLoc='center', bbox=[0.25, -0.5, 0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        # Adjust layout to make room for the table:
        #ax_h.tick_params(axis='x', pad=20)

        #fig_h.subplots_adjust(left=0.2, bottom=0.5)
        #fig_h.tight_layout()

#        save_dir = "../simulation_analysis_saves/Tables/{}/{}/{}/num_sims={}/".format(outcome, iseffect, include_stderr, num_sims)
        save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig_h.savefig(save_dir + "/condition_prop_n={}.png".format(num_steps), bbox_inches = 'tight')
        fig_h.clf()

def hist_means_bias(df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     to_check_ts = None, mean_key = "mean_1"):
    '''
    Not using bias
    '''
    if load_df == True:
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)

        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
            

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_mean1_eg0pt1 = df_for_num_steps_eg0pt1[mean_key]
        df_for_num_steps_mean1_eg0pt3 = df_for_num_steps_eg0pt3[mean_key]
        df_for_num_steps_mean1_unif = df_for_num_steps_unif[mean_key]
        df_for_num_steps_mean1_ts = df_for_num_steps_ts[mean_key]
        
        ax[i].hist(df_for_num_steps_mean1_eg0pt1, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_eg0pt1),2), round(np.var(df_for_num_steps_mean1_eg0pt1), 3)))
        ax[i].hist(df_for_num_steps_mean1_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_eg0pt3),2), round(np.var(df_for_num_steps_mean1_eg0pt3), 3)))

        ax[i].hist(df_for_num_steps_mean1_unif, normed = False, alpha = 0.5, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_unif),2), round(np.var(df_for_num_steps_mean1_unif), 3)))
        ax[i].hist(df_for_num_steps_mean1_ts, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_ts),2), round(np.var(df_for_num_steps_mean1_ts), 3)))

#        ax[i].hist(df_for_num_steps_mean1_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3")
#        ax[i].hist(df_for_num_steps_mean1_unif, normed = False, alpha = 0.5, label = "Uniform")
#        ax[i].hist(df_for_num_steps_mean1_ts, normed = False, alpha = 0.5, label = "Thompson Sampling")
        
        mean_num = int(mean_key.split("_")[-1])
        ax[i].set_xlabel("Mean {} ($\hatp_{}$ with MLE) for number of participants = {} = {}".format(mean_num, mean_num, size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,num_sims)
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")

#        save_dir = "../simulation_analysis_saves/Tables/{}/{}/{}/num_sims={}/".format(outcome, iseffect, include_stderr, num_sims)
    save_dir_ne =  "../simulation_analysis_saves/{}_hist/NoEffect/".format(mean_key)
    save_dir_e =  "../simulation_analysis_saves/{}_hist/Effect/".format(mean_key)
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)


    save_str_ne = save_dir_ne + "/{}.png".format(title) 
    save_str_e = save_dir_e + "/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/{}_hist/Effect/{}.png".format(mean_key, title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()

def hist_means_diff(df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, \
                     title = None,\
                     df_ts = None):
    '''
    Not using bias
    '''
      

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_diff_eg0pt1 = np.abs(df_for_num_steps_eg0pt1["mean_1"] - df_for_num_steps_eg0pt1["mean_2"])
        df_for_num_steps_diff_eg0pt3 = np.abs(df_for_num_steps_eg0pt3["mean_1"] - df_for_num_steps_eg0pt3["mean_2"])


        df_for_num_steps_diff_unif = np.abs(df_for_num_steps_unif["mean_1"] - df_for_num_steps_unif["mean_2"])
        df_for_num_steps_diff_ts = np.abs(df_for_num_steps_ts["mean_1"] - df_for_num_steps_ts["mean_2"])
        
        ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3)), color = "yellow")
        ax[i].hist(df_for_num_steps_diff_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt3),2), round(np.var(df_for_num_steps_diff_eg0pt3), 3)), color = "brown")

        ax[i].hist(df_for_num_steps_diff_unif, normed = False, alpha = 0.5, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_unif),2), round(np.var(df_for_num_steps_diff_unif), 3)), color = "red")
        ax[i].hist(df_for_num_steps_diff_ts, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.var(df_for_num_steps_diff_ts), 3)), color = "blue")

        ax[i].set_xlabel("Difference in Mean Estimates ($\hatp_1$ - $\hatp_2$ with MLE) for number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,num_sims)
        ax[i].set_ylabel("Number of Simulations")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/diff_hist/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/diff_hist/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)


    save_str_ne = save_dir_ne + "/{}.png".format(title) 
    save_str_e = save_dir_e + "/{}.png".format(title) 

#    save_str_ne = "../simulation_analysis_saves/diff_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/diff_hist/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()

def hist_cond1(df_eg0pt1 = None, df_eg0pt3 = None,\
               df_unif = None, df_ts = None, df_tsppd = None, df_ets = None,\
               n = None, num_sims = None,\
                     title = None):
    '''

    '''
      

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]
        df_for_num_steps_tsppd = df_tsppd[df_tsppd['num_steps'] == num_steps]
        df_for_num_steps_ets = df_ets[df_ets['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_diff_eg0pt1 = df_for_num_steps_eg0pt1["sample_size_1"]/num_steps
        df_for_num_steps_diff_eg0pt3 = df_for_num_steps_eg0pt3["sample_size_1"]/num_steps


        df_for_num_steps_diff_unif = df_for_num_steps_unif["sample_size_1"]/num_steps
        df_for_num_steps_diff_ts = df_for_num_steps_ts["sample_size_1"]/num_steps
        df_for_num_steps_diff_tsppd = df_for_num_steps_tsppd["sample_size_1"]/num_steps
        df_for_num_steps_diff_ets = df_for_num_steps_ets["sample_size_1"]/num_steps




        alpha = 0.6 
#        ax[i].hist(df_for_num_steps_diff_unif, normed = False, alpha = alpha, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_unif),2), round(np.var(df_for_num_steps_diff_unif), 3)), color = "red")
#        ax[i].hist(df_for_num_steps_diff_ts, normed = False, alpha = alpha, label = "Thompson Sampling: mean = {} std = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.std(df_for_num_steps_diff_ts), 3)), color = "blue")
#        ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = False, alpha = alpha, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3)), color = "yellow")
 #       ax[i].hist(df_for_num_steps_diff_eg0pt3, normed = False, alpha = alpha, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt3),2), round(np.var(df_for_num_steps_diff_eg0pt3), 3)), color = "green")

        binwidth = 0.1
        bins=np.arange(0, 1 + binwidth, binwidth)

        ax[i].hist(df_for_num_steps_diff_tsppd, normed = False, alpha = alpha, label = "PostDiff TS 0.1: mean = {} std = {}".format(round(np.mean(df_for_num_steps_diff_tsppd),2), round(np.std(df_for_num_steps_diff_tsppd), 3)), color = "purple", bins = bins)
        ax[i].hist(df_for_num_steps_diff_ets, normed = False, alpha = alpha, label = "Epsilon TS 0.1: mean = {} std = {}".format(round(np.mean(df_for_num_steps_diff_ets),2), round(np.std(df_for_num_steps_diff_ets), 3)), color = "green", bins = bins)


        ax[i].set_xlabel("Proportion of Samples in Condition 1" + " for number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,num_sims)
        ax[i].set_ylabel("Number of Simulations")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/cond1_hist/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/cond1_hist/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)


    save_str_ne = save_dir_ne + "/{}.png".format(title) 
    save_str_e = save_dir_e + "/{}.png".format(title) 


#    save_str_ne = "../simulation_analysis_saves/imba_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/imba_hist/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()


def hist_imba(df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, \
                     title = None,\
                     df_ts = None):
    '''
    Not using bias
    '''
      

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_diff_eg0pt1 = np.abs(df_for_num_steps_eg0pt1["sample_size_1"]/num_steps - 0.5)
        df_for_num_steps_diff_eg0pt3 = np.abs(df_for_num_steps_eg0pt3["sample_size_1"]/num_steps - 0.5)


        df_for_num_steps_diff_unif = np.abs(df_for_num_steps_unif["sample_size_1"]/num_steps - 0.5)
        df_for_num_steps_diff_ts = np.abs(df_for_num_steps_ts["sample_size_1"]/num_steps - 0.5)




        alpha = 0.6 
        ax[i].hist(df_for_num_steps_diff_unif, normed = False, alpha = alpha, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_unif),2), round(np.var(df_for_num_steps_diff_unif), 3)), color = "red")
        ax[i].hist(df_for_num_steps_diff_ts, normed = False, alpha = alpha, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.var(df_for_num_steps_diff_ts), 3)), color = "blue")
        ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = False, alpha = alpha, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3)), color = "yellow")
        ax[i].hist(df_for_num_steps_diff_eg0pt3, normed = False, alpha = alpha, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt3),2), round(np.var(df_for_num_steps_diff_eg0pt3), 3)), color = "green")


        ax[i].set_xlabel("Sample Size Imbalance (|$\\frac{n_1}{n}$ - 0.5|)" + " for number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,num_sims)
        ax[i].set_ylabel("Number of Simulations")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/imba_hist/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/imba_hist/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)


    save_str_ne = save_dir_ne + "/{}.png".format(title) 
    save_str_e = save_dir_e + "/{}.png".format(title) 


#    save_str_ne = "../simulation_analysis_saves/imba_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/imba_hist/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()






def hist_assignprob(df = None, ts_ap_df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     to_check_ts = None, mean_key_of_interest = "mean_1", mean_key_other = "mean_2"):
    '''
Compute assign prob for arm for EG
    '''
    #ts_ap = pd.read_csv(ts_ap_df)
    if load_df == True:
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)

        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)

 #    ipdb.set_trace()                                      
    rectify_vars_noNa(df_eg0pt1, alg_key = "TS")
    rectify_vars_noNa(df_eg0pt3, alg_key = "TS")
    rectify_vars_noNa(df_ts, alg_key = "TS")


    of_interest_idx = int(mean_key_of_interest.split("_")[-1])
    other_idx = int(mean_key_other.split("_")[-1])
    #print(data)
    fig_h, ax_h = plt.subplots(2,2) 
    fig_h.set_size_inches(14.5, 10.5)
    ax_h = ax_h.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

#    ipdb.set_trace()
    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]
        ts_ap = ts_ap_df[str(num_steps)]
        if mean_key_of_interest == 2:
            ts_ap = 1 - ts_ap


       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_mean_of_interest_eg0pt1 = df_for_num_steps_eg0pt1[mean_key_of_interest].to_numpy()
        df_for_num_steps_mean_of_interest_eg0pt3 = df_for_num_steps_eg0pt3[mean_key_of_interest].to_numpy()

        df_for_num_steps_mean_other_eg0pt1 = df_for_num_steps_eg0pt1[mean_key_other].to_numpy()
        df_for_num_steps_mean_other_eg0pt3 = df_for_num_steps_eg0pt3[mean_key_other].to_numpy()

        assign_prob_0pt1 = assign_prob_eg_action(arm_of_interest_mean_reward = df_for_num_steps_mean_of_interest_eg0pt1,\
             arm_other_mean_reward = df_for_num_steps_mean_other_eg0pt1, epsilon = 0.1)   

        assign_prob_0pt3 = assign_prob_eg_action(arm_of_interest_mean_reward = df_for_num_steps_mean_of_interest_eg0pt3,\
             arm_other_mean_reward = df_for_num_steps_mean_other_eg0pt3, epsilon = 0.3)                                       

        ax_h[i].hist(assign_prob_0pt1, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(assign_prob_0pt1),  2), round(np.var(assign_prob_0pt1), 3)))
        ax_h[i].hist(assign_prob_0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(assign_prob_0pt3) ,2), round(np.var(assign_prob_0pt3) , 3)))
        ax_h[i].hist(ts_ap, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(ts_ap) ,2), round(np.var(ts_ap) , 3)))

      #  ax[i].hist(df_for_num_steps_mean1_unif, normed = False, alpha = 0.5, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_unif),2), round(np.var(df_for_num_steps_mean1_unif), 3)))
      #  ax[i].hist(df_for_num_steps_mean1_ts, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_mean1_ts),2), round(np.var(df_for_num_steps_mean1_ts), 3)))

#        ax[i].hist(df_for_num_steps_mean1_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3")
#        ax[i].hist(df_for_num_steps_mean1_unif, normed = False, alpha = 0.5, label = "Uniform")
#        ax[i].hist(df_for_num_steps_mean1_ts, normed = False, alpha = 0.5, label = "Thompson Sampling")
        
        ax_h[i].set_xlabel("Assignment probability for arm {} for number of participants = {} = {}".format(of_interest_idx, size_vars[i], num_steps))
        ax_h[i].legend()
        ax_h[i].set_ylim(0,num_sims)
        ax_h[i].set_ylabel("Number of Simulations")
        i +=1  
#    fig_h.subplots_adjust(top=0.5)
    fig_h.suptitle(title)
#    fig.tight_layout(rect=[0, 0.03, 1, 0.80])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/assign_prob{}_hist/NoEffect/".format(of_interest_idx)
    save_dir_e =  "../simulation_analysis_saves/assign_prob{}_hist/Effect/".format(of_interest_idx)
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)


    save_str_ne = save_dir_ne + "/{}.png".format(title) 
    save_str_e = save_dir_e + "/{}.png".format(title) 

 #   save_str_ne = "../simulation_analysis_saves/assign_prob{}_hist/NoEffect/{}.png".format(of_interest_idx, title) 
 #   save_str_e = "../simulation_analysis_saves/assign_prob{}_hist/Effect/{}.png".format(of_interest_idx, title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig_h.savefig(save_str_ne,bbox_inches = 'tight')
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig_h.savefig(save_str_e, bbox_inches = 'tight')

      #plt.show()
    plt.close(fig_h)
    fig_h.clf()

def assign_prob_eg_action(arm_of_interest_mean_reward, arm_other_mean_reward, epsilon):
   """
   comptue assign prob for arm of interest, returns array of length num sims
   
   """
   i = 0
   assign_prob_list = []
   for i in range(len(arm_of_interest_mean_reward)):
        if arm_of_interest_mean_reward[i] > arm_other_mean_reward[i]:
           assign_prob = (1-epsilon)*1 + epsilon*0.5 # 0.9 + 0.05
        else:
            assign_prob = (1 - epsilon)*0 + epsilon*0.5# 0.1*0.5 = 0.05
        assign_prob_list.append(assign_prob)
   return np.array(assign_prob_list)  



def calculate_assgn_prob_by_step_size(actions_root, num_samples, num_actions = 2, cached_probs={}, 
                  prior = [1,1], binary_rewards = True, \
                  config = {}, n = None,\
                  num_sims = None, batch_size = None, no_effect = True, effect_size = None):
    """
    Computes assignment probabilities for ts, sets these to column 'ProbAction{}IsBest'
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
        n = np.ceil(nobs_total)
        step_sizes = [np.ceil(n/2), n, 2*n, 4*n]
    
   # fig, ax = plt.subplots(1,4)
    #ax = ax.ravel()
    
    i=0
    if os.path.isfile(actions_root + "-ThompsonSamplingAP.csv"):
        probs_df = pd.read_csv(actions_root + "-ThompsonSamplingAP.csv")
        print("TS AP dict save found, loading..")
        return probs_df
    else:
        print("no AP save found, creating..")
    #    print("assign prob cache exists at", actions_root + "/withprob")
   # else:
    #    print("no assing prob cache found, computing assing prob for TS...")
    probs_dict = {}
    for num_steps in step_sizes:
        num_steps = int(num_steps)
        probs_per_sim_action_1 = []
        for sim_count in range(num_sims):
            # print(sim_count)
            
            actions_infile = actions_root + "/tbb_actions_{}_{}.csv".format(int(num_steps), sim_count)
            actions_df = pd.read_csv(actions_infile,skiprows=1)
            max_weights = 0
            
            # print(actions_df)
            if binary_rewards:
                all_models, cache_keys = create_models_binary(actions_df, prior, num_actions)
                
            
            else:
                all_models, cache_keys = create_models_normal(actions_df, prior, num_actions)
                
            final_model_idx = len(all_models[0]) - 1 - 1#extra -1 for idx
            final_models = [models[final_model_idx] for models in all_models] #plural for arms
            #print("final_model_idx", final_model_idx)
            if os.path.isdir(actions_root + "/withprob") and 0:
        #        print("assign prob cache exists at", actions_root + "/withprob")
                actions_df_withprob = pd.read_csv(actions_root + "/withprob/tbb_actions_{}_{}_withprob.csv".format(num_steps, sim_count))
                prob = actions_df_withprob[H_ALGO_PROB_BEST_ACTION.format(1)][0]
        #        print(actions_root + "/withprob/tbb_actions_{}_{}_withprob.csv".format(num_steps, sim_count))
       #         print("prob", prob)
            else:
       
         #       print("no assing prob cache found, computing assing prob for TS...")
                counts = thompson_policy.estimate_probability_condition_assignment(None, num_samples, num_actions, final_models)
                probs = [count / num_samples for count in counts]
            
            #condition_assigned = int(actions_df.iloc[final_model_idx].loc[H_ALGO_ACTION])
                prob = probs[0] # map back to 0 indexing, choose Action1
            #print("prob:", prob)
                actions_df[H_ALGO_PROB_BEST_ACTION.format(1)] = prob
                actions_df[H_ALGO_PROB_BEST_ACTION.format(2)] = 1 - prob
            
                if not os.path.isdir(actions_root + "/withprob"):
                    os.mkdir(actions_root + "/withprob")
                actions_df.to_csv(actions_root + "/withprob/tbb_actions_{}_{}_withprob.csv".format(num_steps, sim_count), index=False)
            probs_per_sim_action_1.append(prob)
            

                #probs_per_sim_action_1 = np.array(probs_per_sim_action_1)
                #np.save(, probs_per_sim_action_1)
       #        y, x, _ = ax[i].hist(probs_per_sim_action_1)
       #        ax[i].clear
        probs_dict[str(num_steps)] = probs_per_sim_action_1
#        ax[i].hist(probs_per_sim_action_1)
#        ax[i].set_xlabel(str(num_steps))
#       #print(np.max(np.bincount(np.array(probs_per_sim_action_1))))
#        ax[i].set_ylim(0.0, 255)
#        prop_sims = np.round((np.array(probs_per_sim_action_1) >  0.90).sum()/num_sims + (np.array(probs_per_sim_action_1) <  1-0.90).sum()/num_sims, 3)
#        ax[i].set_title("prop. > 0.90 \n or < 0.10 = {}".format(str(prop_sims)), fontsize = 8.0)
        #ax.text(0, 0.1*max_plot_val, 'Mean = %s' % np.round(mean_outcomes[0],3), ha='center', va='bottom', fontweight='bold', fontsize = 16)
        i = i+1
            #print("proportion of simulations exceeding 0.95 = ", (np.array(probs_per_sim_action_1) > 0.95).sum()/num_sims)
            #print("proportion of simulations below 0.05 = ", (np.array(probs_per_sim_action_1) <  1-0.95).sum()/num_sims)
            #print("proportion of simulations exceeding 0.90 or below 0.10", )
        
   # fig.tight_layout()
#    if no_effect:
#        title = "Distrubtion of Assignment Probability for Acton 1 Across {} Sims \n Batch Size = {}".format(num_sims, batch_size)
#        fig.suptitle(title, y = 1.07)
#        fig.text(0.5, 0.0, "number of participants = n/2, n, 2*n, 4*n for n = {}".format(n), ha='center')
#        
#    else:
#       title = "Distrubtion of Assignment Probability for Acton 1 Across {} Sims \n Batch Size = {} \n Effect Size = {}".format(num_sims, batch_size, effect_size)
#       fig.suptitle(title, y = 1.16)
#       fig.text(0.5, 0.0, "number of participants = n/2, n, 2*n, 4*n for n = {} (n is required for 0.8 power)".format(n), ha='center')
#    #fig.tight_layout()
# 
#
#        
#    #fig.tight_layout()
#    #fig.subplots_adjust(wspace = 0.90)
#    #fig.tight_layout()
#    title = title + "n = {}".format(n)
#    print("saving debug plots!!!")
#    fig.savefig("plots/" + title + ".png")
#  
    #fig.clf()
    df_ap = pd.DataFrame(probs_dict)
    print(df_ap)
    df_ap.to_csv(actions_root + "-ThompsonSamplingAP.csv")
    return df_ap
 

def hist_wald(df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     to_check_ts = None):
    '''
    Not using bias
    '''
    if load_df == True:
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)

        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
   # SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2)
      

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_diff_eg0pt1 = df_for_num_steps_eg0pt1["wald_type_stat"].dropna()
        df_for_num_steps_diff_eg0pt3 = df_for_num_steps_eg0pt3["wald_type_stat"].dropna()


        df_for_num_steps_diff_unif = df_for_num_steps_unif["wald_type_stat"].dropna()
        df_for_num_steps_diff_ts = df_for_num_steps_ts["wald_type_stat"].dropna()
        num_rejected_eg0pt1 = np.sum(df_for_num_steps_diff_eg0pt1 > 1.96) + np.sum(df_for_num_steps_diff_eg0pt1 < -1.96)
        prop_rejected_eg0pt1 = num_rejected_eg0pt1 / num_replications 

        num_rejected_ts = np.sum(df_for_num_steps_diff_ts > 1.96) + np.sum(df_for_num_steps_diff_ts < -1.96)
        prop_rejected_ts = num_rejected_ts / num_replications                                                       

        ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {} \n prop rej. = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3), prop_rejected_eg0pt1))
        ax[i].hist(df_for_num_steps_diff_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt3),2), round(np.var(df_for_num_steps_diff_eg0pt3), 3)))

        ax[i].hist(df_for_num_steps_diff_unif, normed = False, alpha = 0.5, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_unif),2), round(np.var(df_for_num_steps_diff_unif), 3)))
        ax[i].hist(df_for_num_steps_diff_ts, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {} \n prop rej. = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.var(df_for_num_steps_diff_ts), 3), prop_rejected_ts))
        
        ax[i].set_xlabel("Wald Statistic for number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,num_sims)
        ax[i].set_ylabel("Number of Simulations")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_dir_ne =  "../simulation_analysis_saves/wald_hist/NoEffect/"
    save_dir_e =  "../simulation_analysis_saves/wald_hist/Effect/"
    Path(save_dir_ne).mkdir(parents=True, exist_ok=True)
    Path(save_dir_e).mkdir(parents=True, exist_ok=True)

    save_str_ne = save_dir_ne + "{}.png".format(title)
    save_str_e = save_dir_e + "{}.png".format(title)

 #   save_str_ne = "../simulation_analysis_saves/wald_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "../simulation_analysis_saves/wald_hist/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()

def KDE_wald(df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None,\
                     to_check_ts = None):
    '''
    Not using bias
    '''
    if load_df == True:
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)

        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
   # SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2)
      

    #print(data)
    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)
        



        num_replications = len(df_for_num_steps_eg0pt1)

        df_for_num_steps_diff_eg0pt1 = df_for_num_steps_eg0pt1["wald_type_stat"].dropna()
        df_for_num_steps_diff_eg0pt3 = df_for_num_steps_eg0pt3["wald_type_stat"].dropna()


        df_for_num_steps_diff_unif = df_for_num_steps_unif["wald_type_stat"].dropna()
        df_for_num_steps_diff_ts = df_for_num_steps_ts["wald_type_stat"].dropna()
        

        bins = np.linspace(-20, 20, 100)
        kde_eg0pt1 = stats.gaussian_kde(df_for_num_steps_diff_eg0pt1)
        kde_eg0pt3 = stats.gaussian_kde(df_for_num_steps_diff_eg0pt3)
        kde_ts = stats.gaussian_kde(df_for_num_steps_diff_ts)
        kde_unif = stats.gaussian_kde(df_for_num_steps_diff_unif)
       # print(df_for_num_steps_diff_eg0pt1)
   #     ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = True, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3)))
    #    ax[i].plot(bins, kde_eg0pt1(bins), label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(kde_eg0pt1(bins)),2), round(np.var(kde_eg0pt1(bins)), 3)))
   #     ax[i].plot(bins, kde_eg0pt3(bins), label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(kde_eg0pt3(bins)),2), round(np.var(kde_eg0pt3(bins)), 3)))
   #     ax[i].plot(bins, kde_unif(bins), label = "Uniform: mean = {} var = {}".format(round(np.mean(kde_unif(bins)),2), round(np.var(kde_unif(bins)), 3)))
        ax[i].plot(bins, kde_ts(bins), label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(kde_ts(bins)),2), round(np.var(kde_ts(bins)), 3)))
        ax[i].hist(df_for_num_steps_diff_ts, bins = 30, normed = True, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.var(df_for_num_steps_diff_ts), 3)))
       

# ax[i].hist(df_for_num_steps_diff_eg0pt1, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.1: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt1),2), round(np.var(df_for_num_steps_diff_eg0pt1), 3)))
       # ax[i].hist(df_for_num_steps_diff_eg0pt3, normed = False, alpha = 0.5, label = "Epsilon Greedy 0.3: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_eg0pt3),2), round(np.var(df_for_num_steps_diff_eg0pt3), 3)))

        #ax[i].hist(df_for_num_steps_diff_unif, normed = False, alpha = 0.5, label = "Uniform: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_unif),2), round(np.var(df_for_num_steps_diff_unif), 3)))
        #ax[i].hist(df_for_num_steps_diff_ts, normed = False, alpha = 0.5, label = "Thompson Sampling: mean = {} var = {}".format(round(np.mean(df_for_num_steps_diff_ts),2), round(np.var(df_for_num_steps_diff_ts), 3)))

        ax[i].set_xlabel("Wald statistic for number of participants = {} = {}".format(size_vars[i], num_steps))
        ax[i].legend()
        ax[i].set_ylim(0,0.4)
        ax[i].set_ylabel("Number of Simulations")
        i +=1  
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
      # if not os.path.isdir("plots"):
      #    os.path.mkdir("plots")
    save_str_ne = "../simulation_analysis_saves/wald_KDE/NoEffect/{}.png".format(title) 
    save_str_e = "../simulation_analysis_saves/wald_KDE/Effect/{}.png".format(title) 
    if "No Effect" in title:
	    print("saving to ", save_str_ne)
	    fig.savefig(save_str_ne)
    elif "With Effect" in title:
	    print("saving to ", save_str_e)
	    fig.savefig(save_str_e)

      #plt.show()
    plt.clf()
    plt.close()
    

