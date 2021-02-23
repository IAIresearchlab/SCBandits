import matplotlib
matplotlib.use('Agg')
#matplotlib.use("gtk")
#matplotlib.use('Qt5Agg')
from table_functions import *
import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import sys

# print(data)
import numpy as np
import os
from scipy import stats
from pathlib import Path

from matplotlib.pyplot import figure
import glob
import numpy as np
from hist_functions import *
import scipy.stats
import ipdb
from scatter_plot_functions import *
from rectify_vars_and_wald_functions import *
from hypo_testing import * 
from load_cutoffs import *

RL4RL_SECB_DIR = "../../banditalgorithms/src/RL4RLSectionB/"
RL4RL_SECC_DIR = "../../empirical_data/Outfile/ParamBased" #Has sim saves

BF_CUTOFF = 1/3 

SMALL_SIZE = 8#this was 10!!
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def hist_and_cutoffs(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, plot = True, to_check_ipw_mean1 = None, to_check_ipw_mean2 = None, reward_header = None, condition_header = None):
    '''
    TODO rename to_check_ipw to to_check_ipw_wald_stat
    '''
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
    #print(data)
    
    #step_sizes = df['num_steps'].unique()
    step_sizes = [n]
    #size_vars = ["n/2", "n", "2*n", "4*n"]
    size_vars = [n]
    if plot == True:
    #    fig, ax = plt.subplots(2,2)
     #   fig.set_size_inches(14.5, 10.5)
      #  ax = ax.ravel()

        fig, ax = plt.subplots()
        fig.set_size_inches(14.5, 10.5)
        #ax = ax.ravel()
    i = 0
    percenticle_dict_left = {}
    percentile_dict_right = {}



    for num_steps in step_sizes:
        df_unif_for_num_steps = df_unif[df_unif['num_steps'] == num_steps]
        df_unif_for_num_steps_wald = df_unif_for_num_steps['wald_type_stat']
        df_for_num_steps = df[df['num_steps'] == num_steps]

        arm_1_rewards, arm_2_rewards = read_pcrs_rewards(reward_file, reward_header, condition_header) #true rewards

        p1 = sum(arm_1_rewards)/len(arm_1_rewards)
        p2 = sum(arm_2_rewards)/len(arm_2_rewards)
        print(p1, p2)
        delta = np.abs(p1 - p2)

        if to_check_ipw != None:
            to_check_ipw_f = to_check_ipw.format(num_steps)
        
            wald_ipw_per_sim = np.load(to_check_ipw_f)
            ipw_mean_1_per_sim = np.load(to_check_ipw_mean1)
            ipw_mean_2_per_sim = np.load(to_check_ipw_mean2) #E[p_hat_mle]
            
            ipw_mean1 = np.mean(ipw_mean_1_per_sim)
            ipw_mean2 = np.mean(ipw_mean_2_per_sim)
            ipw_var1 = np.var(ipw_mean_1_per_sim)
            ipw_var2 = np.var(ipw_mean_2_per_sim)

            #---------------------TS IPW
            mean_1 = np.load(to_check_ipw_mean1)
            mean_2 = np.load(to_check_ipw_mean2)
            sample_size_1 = df_for_num_steps["sample_size_1"]
            sample_size_2 = df_for_num_steps["sample_size_2"]

            SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
            wald_type_stat_ipw = ((mean_1 - mean_2) - delta)/SE #(P^hat_A - P^hat_b)/SE

            #--------------------TS IPW
            
        


        #---------------------TS delta
        mean_1 = df_for_num_steps["mean_1"]
        mean_2 = df_for_num_steps["mean_2"]
        sample_size_1 = df_for_num_steps["sample_size_1"]
        sample_size_2 = df_for_num_steps["sample_size_2"]
        
        #delta=0
        SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
        wald_type_stat = ((mean_1 - mean_2) - delta)/SE #(P^hat_A - P^hat_b)/SE #Don't subtract delta if simulated for cutoffs delta=0

        #--------------------TS delta





        #---------------------unif delta
        mean_1 = df_unif_for_num_steps["mean_1"]
        mean_2 = df_unif_for_num_steps["mean_2"]
        sample_size_1 = df_unif_for_num_steps["sample_size_1"]
        sample_size_2 = df_unif_for_num_steps["sample_size_2"]

        #delta = 0
        SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
        wald_type_stat_unif = ((mean_1 - mean_2 - delta))/SE #(P^hat_A - P^hat_b)/SE  

        #--------------------unif delta      


        df_unif_for_num_steps = df_unif[df_unif['num_steps'] == num_steps]
        df_unif_for_num_steps_wald = df_unif_for_num_steps['wald_type_stat']
        df_for_num_steps = df[df['num_steps'] == num_steps]
        mle_mean1 = np.mean(df_for_num_steps['mean_1'])
        mle_mean2 = np.mean(df_for_num_steps['mean_2'])

        mle_var1 = np.var(df_for_num_steps['mean_1'])
        mle_var2 = np.var(df_for_num_steps['mean_2'])


        unif_mean1 = np.mean(df_unif_for_num_steps['mean_1'])
        unif_mean2 = np.mean(df_unif_for_num_steps['mean_2'])

        mle_var1_unif = np.var(df_unif_for_num_steps['mean_1'])
        mle_var2_unif = np.var(df_unif_for_num_steps['mean_2'])
        
        
        df_wald_type_per_sim = df_for_num_steps['wald_type_stat']
      #  df_unif_for_num_steps = np.ma.masked_invalid(df_unif_for_num_steps)
        #print(np.mean(df_unif_for_num_steps))
       
        if plot == True:
            #ax[i].hist(df_unif_for_num_steps, density = True)
#changed from Uniform for debugging
            ax.hist(wald_type_stat_unif, normed = True, alpha = 0.5, \
              label = "Uniform: \n$\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} var($\hatp_2$) = {}".format(
                      np.round(np.mean(wald_type_stat_unif), 3),\
                      np.round(np.std(wald_type_stat_unif), 3), np.round(unif_mean1 - p1, 3), np.round(unif_mean2 - p2, 3), np.round(mle_var1_unif,3), np.round(mle_var2_unif,3)
                      )
              )
            if to_check_ipw != None:

                ax.hist(wald_type_stat_ipw, \
                  normed = True, alpha = 0.5,\
                  label = "\n IPW: \n $\mu$ = {} \n$\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} var($\hatp_2$) = {}".format(
                          np.round(np.mean(wald_type_stat_ipw), 3), \
                          np.round(np.std(wald_type_stat_ipw), 3), \
                          np.round(ipw_mean1 - p1,3), np.round(ipw_mean2 - p2,3), np.round(ipw_var1,3), np.round(ipw_var2,3)
                          )
                  )
           
            ax.hist(wald_type_stat, \
              normed = True, alpha = 0.5, \
              label = "\n TS PCRS data: \n $\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} var($\hatp_2$) = {}".format(
                      np.round(np.mean(wald_type_stat), 3), \
                      np.round(np.std(wald_type_stat), 3), \
                      np.round(mle_mean1 - p1,3), np.round(mle_mean2 - p2,3), np.round(mle_var1,3), np.round(mle_var2,3)
                      )
              )
            ax.set_xlabel("number of participants = {} = {}".format(size_vars[i], num_steps))
            ax.axvline(x = np.percentile(wald_type_stat, 2.5), linestyle = "--", color = "black")
            ax.axvline(x = np.percentile(wald_type_stat, 97.5), linestyle = "--", color = "black")
            ax.text(np.percentile(wald_type_stat, 2.5), 0.35, np.round(np.percentile(wald_type_stat, 2.5), 3))
            ax.text(np.percentile(wald_type_stat, 97.5), 0.35, np.round(np.percentile(wald_type_stat, 97.5),3))

            ax.axvline(x = np.percentile(wald_type_stat_unif, 2.5), linestyle = "--", color = "red")
            ax.axvline(x = np.percentile(wald_type_stat_unif, 97.5), linestyle = "--", color = "red")
#            ax[i].text(0.85, 0.5,'Mean = {}, Std = {}'.format(np.mean(df_wald_type_per_sim), np.std(df_wald_type_per_sim)),
#             horizontalalignment='center',
#             verticalalignment='center',
#             transform = ax[i].transAxes)
            
         #   ax[i]
            
            
            
            mu = 0
            variance = 1
            sigma = np.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma))
            ax.legend()
            
            #print("mean, std", np.mean(df_wald_type_per_sim), np.std(df_wald_type_per_sim))
              
        
        percenticle_dict_left[str(num_steps)] = np.percentile(wald_type_stat, 2.5)
        percentile_dict_right[str(num_steps)] = np.percentile(wald_type_stat, 97.5)
        
        i+=1    
    if plot == True:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.80])
        #if not os.path.isdir("Histograms"):
         #   os.path.mkdir("plots")
        print("saving to ", "Histograms/{}.png".format(title))
        fig.savefig("Histograms/{}.png".format(title))

        plt.show()
        plt.clf()
        plt.close()
    
    return percenticle_dict_left, percentile_dict_right

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
        fig_h.savefig("../simulation_analysis_saves/histograms/ExploreAndExploit/N={}/condition_prop_n={}.png".format(n, num_steps), bbox_inches = 'tight')
        fig_h.clf()



def stacked_bar_plot_with_cutoff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, df_ts_jeff = None,\
                     title = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, epsilon = None, es = None, arm_prob = None, reward_table = None, prop1_table = None):
    
    step_sizes = df_ts['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []
    t1_list_ts_welch = []
    t1_list_ts_bf = []
    t1_list_ts_ipw = []

    steps_req_pwer = [785, 88]

    for num_steps in step_sizes:

        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps].dropna()
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps].dropna()
       # ipdb.set_trace()

        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()

#        if num_steps == 785:
#            ipdb.set_trace()
        df_for_num_steps_ts = df_for_num_steps_ts[0:5000]


        df_for_num_steps_ts_jeff = df_ts_jeff[df_ts_jeff['num_steps'] == num_steps].dropna()
        df_for_num_steps_ts_jeff = df_for_num_steps_ts_jeff[0:5000]

        cut_off_ap_dict_ts = {}
        cut_off_ap_dict_ts_jeff = {}

        ts_cutoffs, ts_cutoffs_jeff = load_cutoffs(n = n, num_steps = num_steps, arm_prob = 0.5)
        cut_off_ap_dict_ts["ap0.5"] = ts_cutoffs
        cut_off_ap_dict_ts_jeff["ap0.5"] = ts_cutoffs_jeff

        ts_cutoffs, ts_cutoffs_jeff = load_cutoffs(n = n, num_steps = num_steps, arm_prob = 0.25)
        cut_off_ap_dict_ts["ap0.25"] = ts_cutoffs
        cut_off_ap_dict_ts_jeff["ap0.25"] = ts_cutoffs_jeff

        index, ts_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_ts, cutoffs = cut_off_ap_dict_ts, n=n, es =es)
        index, ts_jeff_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_ts_jeff, cutoffs = cut_off_ap_dict_ts_jeff, n=n, es =es)

        index, unif_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_unif, use_ipw = False, n=n, es =es)
        index, eg0pt1_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_eg0pt1, use_ipw = False, n=n, es =es)

        columns_data = zip(ts_t1_list_all_tests, ts_jeff_t1_list_all_tests, unif_t1_list_all_tests, eg0pt1_t1_list_all_tests)
        columns = ["Thompson Sampling","Thompson Sampling Jefferey's Prior", "Uniform Random", "Epsilon Greedy 0.1"]

        df = pd.DataFrame(columns_data, columns =columns, index = index) 

        anna_table_save_dir = "../simulation_analysis_saves/Tables/Anna_spec/FPR/Arm_Prob={}/Effect_size={}/".format(arm_prob, es)
        Path(anna_table_save_dir).mkdir(parents=True, exist_ok=True)

        save_file = anna_table_save_dir + "num_steps={}.csv".format(num_steps)
        print("saving to ", save_file)
        df.to_csv(save_file)

        if num_steps in steps_req_pwer:
            df_list = [df_for_num_steps_eg0pt1, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_ts_jeff]
            prop1_table_col_final = get_prop_majority_col(df_list)
            reward_table_col_final = get_alg_reward_col(df_list)

            #tests_list = np.array([t1_wald, t1_welch, t1_bf, t1_ipw, t1_simbased_025, t1_simbased_05])
            reward_table.append(reward_table_col_final)
            prop1_table.append(prop1_table_col_final)
        #--------------------Done Anna Spec Tables---------------------
        #--------------------Done Anna Spec Tables---------------------

        num_replications = len(df_for_num_steps_ts)

        #ipdb.set_trace()
        t1_ipw = np.sum(df_for_num_steps_ts["wald_pval_ipw"] < 0.05)/ num_replications

        t1_list_ts_ipw.append(t1_ipw)

        sample_size_1 = df_for_num_steps_ts["sample_size_1"]
        sucesses_1 = df_for_num_steps_ts["mean_1"]*sample_size_1

        sample_size_2 = df_for_num_steps_ts["sample_size_2"]
        sucesses_2 = df_for_num_steps_ts["mean_2"]*sample_size_2

        bf = BF(sucesses_1, sample_size_1, sucesses_2, sample_size_2)
        rejected_bf = np.sum(bf < BF_CUTOFF)#reject H0 if BF is lower than 1
        t1_ts_bf = rejected_bf/num_replications


        t1_list_ts_bf.append(t1_ts_bf)

        pvals_welch = welch_test(means_1 = df_for_num_steps_ts["mean_1"], sample_size_1 = df_for_num_steps_ts["sample_size_1"], means_2 = df_for_num_steps_ts["mean_2"], sample_size_2 = df_for_num_steps_ts["sample_size_2"])
        rejected_welch = np.sum(pvals_welch < 0.05)
        t1_welch = rejected_welch/num_replications
        t1_list_ts_welch.append(t1_welch)
        #df_for_num_steps_unif = df_for_num_steps_unif.dropna()
       # bins = np.arange(0, 1.01, .025)

      #  plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon = epsilon, n=n)

 
       # print(num_replications)

       # ipdb.set_trace()
        num_replications = len(df_for_num_steps_eg0pt1)
        #num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['pvalue'] < .05) #Epsilon Greedy
        num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['wald_pval'] < .05) #Epsilon Greedy
        #num_rejected_eg0pt1 = np.sum(wald_pval_eg0pt1 < .05) #Epsilon Greedy

        #num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['pvalue'] < .05) #Epsilon Greedy
        num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['wald_pval'] < .05) #Epsilon Greedy

        #num_rejected_ts = np.sum(df_for_num_steps_ts['pvalue'] < .05) #Thompson
        num_rejected_ts = np.sum(df_for_num_steps_ts['wald_pval'] < .05) #Thompson

#        num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
        num_rejected_unif = np.sum(df_for_num_steps_unif['wald_pval'] < .05)

        var = np.var(df_for_num_steps_unif['pvalue'] < .05)
        
        num_replications = len(df_for_num_steps_eg0pt1)
        t1_eg0pt1 = num_rejected_eg0pt1 / num_replications
        num_replications = len(df_for_num_steps_eg0pt3)
        t1_eg0pt3 = num_rejected_eg0pt3 / num_replications

        num_replications = len(df_for_num_steps_ts)
        t1_ts = num_rejected_ts / num_replications
        num_replications = len(df_for_num_steps_unif)
        t1_unif =num_rejected_unif / num_replications
       
        t1_list_unif.append(t1_unif)
        t1_list_ts.append(t1_ts)
        
        t1_list_eg0pt1.append(t1_eg0pt1)
        t1_list_eg0pt3.append(t1_eg0pt3)
        var_list.append(var)

        ts_col = [t1_ts, t1_welch, t1_ts_bf, t1_ipw]
        index = ["Wald Test", "Welch Test", "Bayes Factor (Cutoff {})".format(BF_CUTOFF), "IPW"] 
        
       # test_by_alg_table([t1_ts, t1_welch, t1_ts_bf, t1_ipw])
        
    t1_list_ts = np.array(t1_list_ts)
    t1_list_ts_welch = np.array(t1_list_ts_welch)
    t1_list_ts_bf = np.array(t1_list_ts_bf)
    t1_list_ts_ipw = np.array(t1_list_ts_ipw)
    ind = np.arange(3*len(step_sizes), step=3)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    width = 0.56
    width = 0.40
    capsize = width*4
    width_total = 2*width
    
   
    t1_list_eg0pt1 = np.array(t1_list_eg0pt1)
    t1_list_eg0pt3 = np.array(t1_list_eg0pt3)
    t1_list_unif = np.array(t1_list_unif)
    
    t1_eg0pt1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt1*(1-t1_list_eg0pt1)/num_sims) #95 CI for Proportion
    t1_eg0pt3_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt3*(1-t1_list_eg0pt3)/num_sims) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims)
    t1_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts*(1-t1_list_ts)/num_sims)
    t1_se_ts_welch = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts_welch*(1-t1_list_ts_welch)/num_sims)
    t1_se_ts_bf = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts_bf*(1-t1_list_ts_bf)/num_sims)
    #print(t1_se_unif)
    p1 = ax.bar(ind, t1_list_eg0pt1, width = width, yerr = t1_eg0pt1_se, \
                ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
    p3 = ax.bar(ind+width, t1_list_eg0pt3, width = width, yerr = t1_eg0pt3_se, \
                ecolor='black', capsize=capsize, color = 'green', edgecolor='black')
  
    p4 = ax.bar(ind+2*width, t1_list_ts, width = width, yerr = t1_se_ts,     
               ecolor='black', capsize=capsize, color = 'blue', edgecolor='black') 

    p5 = ax.bar(ind+3*width, t1_list_ts_welch, width = width, yerr = t1_se_ts_welch,     
               ecolor='black', capsize=capsize, color = 'brown', edgecolor='black') 

    p6 = ax.bar(ind+4*width, t1_list_ts_bf, width = width, yerr = t1_se_ts_bf,     
               ecolor='black', capsize=capsize, color = 'orange', edgecolor='black') 
  
    p2 = ax.bar(ind-width, t1_list_unif, width = width,\
                 yerr = t1_se_unif, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')
    if ax_idx == 2:
    #   leg1 = ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Epsilon Greedy Chi Squared 0.1', "Uniform Chi Squared", "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))
       leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0], p5[0], p6[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald", "Thompson Sampling Welch", "Thompson Sampling BF"), bbox_to_anchor=(1.0, 1.76))  
       #leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Chi Squared", 'Epsilon Greedy Chi Squared 0.1', "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))  
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    
    ax.set_ylim(0, 1.01)
    ax.axhline(y=0.80, linestyle='--')


    return [t1_list_unif, t1_list_eg0pt1, t1_list_ts, t1_list_ts_ipw] #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def parse_dir(root, root_cutoffs, num_sims):
    arm_prob= 0.5
    arm_prob_list = [0.25, 0.5]
#    arm_prob_list = [0.5]
#    es_list = [0.5, 0.3, 0.1]
    n_list = [32, 88, 785]
#    n_list = [32, 88, 785]
#    ipdb.set_trace()

    arm_prob = 0.5
    for arm_prob in arm_prob_list:
        epsilon = 0.1
    #EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
        root_dir = root + "/num_sims={}armProb={}".format(num_sims, 0.5)
        unif_root = RL4RL_SECB_DIR + "/simulation_saves/UniformIsEffect/num_sims={}armProb={}".format(num_sims, arm_prob)
        unif_root = RL4RL_SECB_DIR + "/simulation_saves/UniformNoEffect/num_sims={}armProb={}".format(num_sims, arm_prob)
        eg_root ="../simulation_saves/EpsilonGreedyNoEffectFast/num_sims={}armProb={}".format(num_sims, arm_prob)
       # unif_root_secC = RL4RL_SECC_DIR + "/simulation_saves/UniformIsEffect/num_sims={}armProb={}".format(num_sims, arm_prob)"/numSims5000problem_print_output/Y2/"
        fig, ax = plt.subplots(1,4, figsize = (12,5))
        #fig.set_size_inches(17.5, 13.5)
        ax = ax.ravel()
        i = 0

        #root_ts = RL4RL_SECB_DIR + "/simulation_saves/IsEffect_fixedbs_RL4RLMay8/num_sims={}armProb=0.5".format(num_sims)
        num_sims_ts = 10000
        root_ts = "../simulation_saves/TSPPDNoEffectResampleFast/num_sims={}armProb={}".format(num_sims_ts, arm_prob)
        root_ts_jeff = "../simulation_saves/TSPPDNoEffectFastJeffPrior/num_sims={}armProb={}".format(num_sims_ts, arm_prob)

        reward_table = []
        prop1_table = []
        for n in n_list:
    #        es = es_list[i]
            bs = 1
            es_dir_0pt1 = eg_root + "/N={}epsilon={}/".format(n, 0.1)
            es_dir_0pt3 = root_dir + "/N={}epsilon={}/".format(n, 0.3)


            ts_dir = root_ts + "/N={}c=0.0/".format(n)
            ts_jeff_dir = root_ts_jeff + "/N={}c=0.0/".format(n)
            unif_dir = unif_root + "/N={}/".format(n)

            to_check_eg0pt1 = glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

          
            to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,n))) == 1)

            to_check_unif = glob.glob(es_dir_0pt1 + "/*Uniform*{}Df*.pkl".format(n))[0]
    #        assert(len(glob.glob(es_dir_0pt1 + "/*Uniform*{}Df.pkl".format(es))) == 1)

    #        bbUnEqualMeansEqualPriorburn_in_size-batch_size=1-1BB0.1Df_sim=5000_m=0_r=0
            to_check_ts= glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

            to_check_ts_jeff = glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)
      

#            if n == 785 and arm_prob == 0.5:
#                ipdb.set_trace()
          #, 657------hists, tables etc, remeber 0:5000 
            with open(to_check_eg0pt1, 'rb') as f:
                df_eg0pt1 = pickle.load(f)
            with open(to_check_eg0pt3, 'rb') as f:
                df_eg0pt3 = pickle.load(f)
                                                   
            with open(to_check_unif, 'rb') as f:
                df_unif = pickle.load(f)

            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
              
            with open(to_check_ts_jeff, 'rb') as t:
                df_ts_jeff = pickle.load(t)
              
           # ipdb.set_trace()
            rect_key = "Drop NA"
            #rect_key = "TS"
            rectify_vars_noNa(df_eg0pt1, alg_key = rect_key)
            rectify_vars_noNa(df_eg0pt3, alg_key = rect_key)
            rectify_vars_noNa(df_ts, alg_key = rect_key)
            rectify_vars_noNa(df_ts_jeff, alg_key = rect_key)
            rectify_vars_noNa(df_unif, alg_key = rect_key)
       
            assert np.sum(df_eg0pt1["wald_type_stat"].isna()) == 0
            assert np.sum(df_eg0pt1["wald_pval"].isna()) == 0
     
            es = 0
            next_df = stacked_bar_plot_with_cutoff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                         df_unif = df_unif, df_ts = df_ts, df_ts_jeff = df_ts_jeff, \
                                                 n = n, num_sims = num_sims,
                                                   ax = ax[i], ax_idx = i, epsilon = epsilon, es = es, arm_prob = arm_prob, reward_table = reward_table, prop1_table = prop1_table)

            ax[i].set_title("Number of Participants = {} ".format(n))
            ax[i].set_ylabel("FPR")
            i += 1

            df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
            df.index = ["Uniform Random Wald","Epsilon Greedy Wald", "Thompson Sampling Wald", "Thompson Sampling IPW Wald"]
            df.to_csv("../simulation_analysis_saves/Tables/Power_n={}_numsims={}.csv".format(n, num_sims)) 

            es = 0
            p1 = 0.5 + es/2
            p2 = 0.5 - es/2
            title_ap1 = "Arm 1 Assignment Probability Distribution For n = {} \n Across {} Simulations \n No Effect \n $p_1$ = $p_2$ = {}".format(n, num_sims, p2)
            title_ap2 = "Arm 2 Assignment Probability Distribution For n = {} \n Across {} Simulations \n No Effect \n $p_1$ = $p_2$ = {}".format(n, num_sims, p2)
#    assign_prob_0pt1, ts_ap 
            assign_prob_0pt1, ts_ap = hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                             to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                           title = title_ap1, \
                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")

           
            title_diff = "Distribution of Difference in Posterior Means n = {} \n Across {} Simulations \n No Effect \n $p_1$ = $p_2$ = 0.5".format(n, num_sims)
            title_pa1g = "Distribution of Posterior Probability that $p_1$ > $p_2$ For n = {} \n Across {} Simulations \n No Effect \n $p_1$ = $p_2$ = 0.5".format(n, num_sims)


            df_for_num_steps_pa1g_ts, df_for_num_steps_pa1g_eg0pt1, df_for_num_steps_pa1g_unif = hist_prob_arm1_grtr(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                         df_unif = df_unif, df_ts = df_ts,\
                                           title = title_pa1g, \
                                           n = n, num_sims = num_sims)
    #        title_diff = "Difference in Means (|$\hatp_1$ - $\hatp_2$| with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

            # Note this is posterior mean difference
            df_for_num_steps_diff_ts, df_for_num_steps_diff_eg0pt1, df_for_num_steps_diff_unif = hist_means_diff_post(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                         df_unif = df_unif, df_ts = df_ts,\
                                           title = title_diff, \
                                           n = n, num_sims = num_sims)
            diff_post_list = [df_for_num_steps_diff_unif, df_for_num_steps_diff_ts, df_for_num_steps_diff_eg0pt1] 
            abs_diff_post_list = [np.abs(el) for el in diff_post_list]
#            ipdb.set_trace()
            ap_list = [np.full(ts_ap.shape, 0.5), ts_ap, assign_prob_0pt1]
            alg_key_list = ["Uniform Random", "Thompson Sampling", "Epsilon Greedy 0.1"]

            for diffs, abs_diffs, aps, alg_key in zip(diff_post_list, abs_diff_post_list, ap_list, alg_key_list):
                save_percentile_table(diffs = diffs, abs_diffs = abs_diffs, aps = aps, n = n, alg_key = alg_key)

            title_wald = "No Effect Histogram of Wald Statistic Across {} Simulaitons".format(num_sims)
            if n == 785:
                hist_wald(df_ts  = df_ts, df_eg0pt1 = df_eg0pt1, df_unif = df_unif, n = n, num_sims = num_sims, title = title_wald)

                summary_table_helper(df_ts = df_ts, df_unif = df_unif,  df_eg0pt1 = df_eg0pt1, effect_size = 0, n = n, num_sims = num_sims)

                title_kde = "Wald Statistic KDE Sampling Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.50".format(n, num_sims)#

                KDE_wald(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                             to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                           title = title_kde, \
                                           n = n, num_sims = num_sims)
#                summary_dict = {k: [] for k in ["mean_1_avg MLE","mean_2_avg MLE","mean_1_se_avg MLE", "mean_2_se_avg MLE"]}   
#                summary_table(df_ts, p1 = 0.5, p2 = 0.5, n = n, summary_dict = summary_dict, alg_key = "TS")
#                summary_table(df_unif, p1 = 0.5, p2 = 0.5, n = n, summary_dict = summary_dict, alg_key = "UR")
#                summary_table(df_eg0pt1, p1 = 0.5, p2 = 0.5, n = n, summary_dict = summary_dict, alg_key = "EG")
#
#                df = pd.DataFrame.from_dict(summary_dict)
#                df = df.round(3)
#                #Must respect ordering
#                df["Algorithm"] = ["TS", "UR", "EG 0.1"]
##                df.index("Algorithm")
#
#                save_dir = "../simulation_analysis_saves/Tables/Summary/NoEffect/n={}/num_sims={}/".format(n, num_sims)
#                Path(save_dir).mkdir(parents=True, exist_ok=True)
#                save_file = save_dir + "n={}_numsims={}.csv".format(n, num_sims)
#
#                df.set_index("Algorithm", inplace = True)
#                df.to_csv(save_file)



        columns = ['Epsilon Greedy 0.1', "Uniform Random", 'Thompson Sampling', "Thompson Sampling Jefferey's Prior"]
        index = ['Effect Size 0 - n = 785', 'Effect Size 0 - n = 88']
        print(reward_table)
        reward_table_df = pd.DataFrame(reward_table, index = index, columns = columns)
        prop1_table_df = pd.DataFrame(prop1_table, index = index, columns = columns)

        anna_table_save_dir_reward = "../simulation_analysis_saves/Tables/Anna_spec/Reward/"
        Path(anna_table_save_dir_reward).mkdir(parents=True, exist_ok=True)

        anna_table_save_dir_prop1 = "../simulation_analysis_saves/Tables/Anna_spec/Prop1/"
        Path(anna_table_save_dir_prop1).mkdir(parents=True, exist_ok=True)

        save_file = anna_table_save_dir_reward + "reward_noeffect.csv"
        print("saving to ", save_file)
        reward_table_df.to_csv(save_file)

        save_file = anna_table_save_dir_prop1 + "prop1_noeffect.csv"
        print("saving to ", save_file)
        prop1_table_df.to_csv(save_file)
        title = "FPR Across {} Simulations".format(num_sims)
                #ax[i].set_title(title, fontsize = 55)
                #i +=1
                #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
        fig.suptitle(title)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        #handles, labels = ax[i-1].get_legend_handles_labels()
        
        #fig.legend(handles, labels, loc='upper right', prop={'size': 50})
            #fig.tight_layout()
        save_dir = "../simulation_analysis_saves/power_t1_plots"
        if not os.path.isdir(save_dir):
              os.mkdir(save_dir)

        #fig.set_tight_layout(True)
        fig.tight_layout()
        fig.subplots_adjust(top=.8)

        fig.savefig(save_dir + "/{}.svg".format(title), bbox_inches = 'tight')
       # plt.show()
        fig.clf()
        plt.close(fig)






        
if __name__ == "__main__":
    root = RL4RL_SECB_DIR + "/simulation_saves/EpsilonGreedyNoEffect"
    #parse_dir(root, root_cutoffs)
    #num_sims = 500
    num_sims = 5000
    parse_dir(root, root, num_sims)


