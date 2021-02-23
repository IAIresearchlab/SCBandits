import matplotlib
matplotlib.use('Agg')
#matplotlib.use("gtk")
#matplotlib.use('Qt5Agg')
from table_functions import *
import pickle
import os
from load_cutoffs import *
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

RL4RL_SECB_DIR = "../../banditalgorithms/src/RL4RLSectionB/"
RL4RL_SECC_DIR = "../../empirical_data/Outfile/ParamBased" #Has sim saves

BF_CUTOFF = 1/3 

SMALL_SIZE = 13
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



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



def stacked_bar_plot_with_cutoff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, num_sims = None, df_ts_jeff = None, df_ts_co = None, df_ts_jeff_co = None, \
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
    steps_req_pwer = [785, 88, 657]

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

        index, ts_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_ts, cutoffs = cut_off_ap_dict_ts)
        index, ts_jeff_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_ts_jeff, cutoffs = cut_off_ap_dict_ts_jeff)

        index, unif_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_unif, use_ipw = False)
        index, eg0pt1_t1_list_all_tests = get_all_hyp_tests_for_algo(df_for_num_steps_eg0pt1, use_ipw = False)

        columns_data = zip(ts_t1_list_all_tests, ts_jeff_t1_list_all_tests, unif_t1_list_all_tests, eg0pt1_t1_list_all_tests)
        columns = ["Thompson Sampling","Thompson Sampling Jefferey's Prior", "Uniform Random", "Epsilon Greedy 0.1"]

        df = pd.DataFrame(columns_data, columns =columns, index = index) 

        anna_table_save_dir = "../simulation_analysis_saves/Tables/Anna_spec/Power/Arm_Prob={}/Effect_size={}/".format(arm_prob, es)
        Path(anna_table_save_dir).mkdir(parents=True, exist_ok=True)

        save_file = anna_table_save_dir + "num_steps={}.csv".format(num_steps)
        print("saving to ", save_file)
        df.to_csv(save_file)

        if num_steps in steps_req_pwer:
           # initialize_es0 = False
           # if len(prop1_table) == 0:
           #     initialize_es0 = True

            df_list = [df_for_num_steps_eg0pt1, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_ts_jeff]
            reward_table_col_final = get_alg_reward_col(df_list)
            prop1_table_col_final = get_prop_majority_col(df_list)

            #tests_list = np.array([t1_wald, t1_welch, t1_bf, t1_ipw, t1_simbased_025, t1_simbased_05])
            reward_table.append(reward_table_col_final)
            prop1_table.append(prop1_table_col_final)
        #--------------------Done Anna Spec Tables---------------------
        


        num_replications = len(df_for_num_steps_ts)

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
        index = ["Wald Test", "Welch Test", "Bayes Factor", "IPW"] 
        
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
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.5, 0.3, 0.1, 0.08]
    n_list = [32, 88, 785, 657/4]
#    n_list = [32, 88, 785]
#    ipdb.set_trace()

    arm_prob = 0.5
    epsilon = 0.1
#EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
    root_dir = root + "/num_sims={}armProb={}".format(num_sims, arm_prob)
    unif_root = RL4RL_SECB_DIR + "/simulation_saves/UniformIsEffect/num_sims={}armProb={}".format(num_sims, arm_prob)
    eg_root ="../simulation_saves/EpsilonGreedyIsEffectFast/num_sims={}armProb={}".format(num_sims, arm_prob)
   # unif_root_secC = RL4RL_SECC_DIR + "/simulation_saves/UniformIsEffect/num_sims={}armProb={}".format(num_sims, arm_prob)"/numSims5000problem_print_output/Y2/"
    fig, ax = plt.subplots(1,4, figsize = (12,5))
    #fig.set_size_inches(17.5, 13.5)
    ax = ax.ravel()
    i = 0

    #root_ts = RL4RL_SECB_DIR + "/simulation_saves/IsEffect_fixedbs_RL4RLMay8/num_sims={}armProb=0.5".format(num_sims)
    root_ts = "../simulation_saves/TSPPDIsEffectFast/num_sims={}armProb=0.5".format(num_sims)
    root_ts_jeff = "../simulation_saves/TSPPDIsEffectFastJeffPrior/num_sims={}armProb=0.5".format(num_sims)

    #for CO--------
    num_sims_ts = 10000
    root_ts_co = "../simulation_saves/TSPPDNoEffectFast/num_sims={}armProb=0.5".format(num_sims_ts)
    root_ts_jeff_co = "../simulation_saves/TSPPDNoEffectFastJeffPrior/num_sims={}armProb=0.5".format(num_sims_ts)
    reward_table = []
    prop1_table = []
    for n in n_list:
        es = es_list[i]
        bs = 1
        es_dir_0pt1 = eg_root + "/es={}epsilon={}/".format(es, 0.1)
        es_dir_0pt3 = root_dir + "/es={}epsilon={}/".format(es, 0.3)
        if es == 0.08:
            es_dir_0pt3 = root_dir + "/es={}epsilon={}/".format(0.1, 0.3)

        ts_dir_co = root_ts_co + "/N={}c=0.0/".format(n)
        ts_jeff_dir_co = root_ts_jeff_co + "/N={}c=0.0/".format(n)

        ts_dir = root_ts + "/es={}c=0.0/".format(es)
        ts_jeff_dir = root_ts_jeff + "/es={}c=0.0/".format(es)
        unif_dir = unif_root + "/es={}/".format(es)

        to_check_eg0pt1 = glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df*.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df*.pkl".format(bs,es))) == 1)

        if es == 0.08:
            to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,0.1))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,0.1))) == 1)
      
        else:
            to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_unif = glob.glob(es_dir_0pt1 + "/*Uniform*{}Df*.pkl".format(es))[0]
#        assert(len(glob.glob(es_dir_0pt1 + "/*Uniform*{}Df.pkl".format(es))) == 1)

#        bbUnEqualMeansEqualPriorburn_in_size-batch_size=1-1BB0.1Df_sim=5000_m=0_r=0
        to_check_ts= glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,es))) == 1)

        to_check_ts_jeff = glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,es))) == 1)
  
  
#        if n == 785 and arm_prob == 0.5:
#            ipdb.set_trace()

#        ipdb.set_trace()
      #, 657------hists, tables etc
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)
                                               
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)

        with open(to_check_ts, 'rb') as t:
            df_ts = pickle.load(t)

#        with open(to_check_ts_co, 'rb') as t:
#            df_ts_co = pickle.load(t)
          
        with open(to_check_ts_jeff, 'rb') as t:
            df_ts_jeff = pickle.load(t)

#        with open(to_check_ts_jeff_co, 'rb') as t:
#            df_ts_jeff_co = pickle.load(t)
          
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
 

        next_df = stacked_bar_plot_with_cutoff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3, \
                                     df_unif = df_unif, df_ts = df_ts, df_ts_jeff = df_ts_jeff, \
                                             n = n, num_sims = num_sims,
                                               ax = ax[i], ax_idx = i, epsilon = epsilon, es = es, arm_prob = arm_prob, reward_table= reward_table, prop1_table = prop1_table)

        p1 = 0.5 + es/2
        p2 = 0.5 - es/2
        title_diff = " Difference in Means (|$\hatp_1$ - $\hatp_2$| with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)


#        title_diff = "Difference in Means (|$\hatp_1$ - $\hatp_2$| with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

        hist_means_diff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                     df_unif = df_unif, df_ts = df_ts,\
                                           title = title_diff, \
                                           n = n, num_sims = num_sims)

#        title_imba = "Sample Size Imbalance (|$\\frac{n_1}{n} - 0.5$|"+" Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

        title_imba = "Sample Size Imbalance (|$\\frac{n_1}{n} - 0.5$|"+"Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)


        hist_imba(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                     df_unif = df_unif, df_ts = df_ts,\
                                           title = title_imba, \
                                           n = n, num_sims = num_sims)

 


        title_table = "TODO"
        #if n == 657/4:
         #   ipdb.set_trace()

        table_means_diff(df_ts = df_ts, df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3, df_unif = df_unif, n = n, num_sims = num_sims, \
                        title = title_table, iseffect="WithEffect")

   #     scatter_correlation_helper_outer(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
   #                                  df_unif = df_unif, df_ts = df_ts,\
    #                                       n = n, num_sims = num_sims)  #Title created in helper fn


    #    title_pval = "Chi Squared P value Disutrbtuion For n = {} \n Across {} Simulations".format(n, num_sims)
        
        
     #   hist_pval(to_check = to_check_eg0pt1, to_check_unif = to_check_unif, to_check_ts = to_check_ts, n = n, num_sims = num_sims, load_df = True, \
      #               title = title_pval, plot = True)

        p1 = 0.5 + es/2
        p2 = 0.5 - es/2
        title_mean1 = "Mean 1 ($\hatp_1$ with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)
        title_mean2 = "Mean 2 ($\hatp_2$ with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)

        hist_means_bias(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                           title = title_mean1, \
                                           n = n, num_sims = num_sims, mean_key = "mean_1")

        hist_means_bias(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                           title = title_mean2, \
                                           n = n, num_sims = num_sims, mean_key = "mean_2")

        

        if n == 785:
            title_kde = "Wald Statistic KDE Sampling Disutrbtuion For n = {} \n Across {} Simulations \n With Effect $p_1$ = 0.55, $p_2$ = 0.45".format(n, num_sims)#

            KDE_wald(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                         to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                           title = title_kde, \
                                           n = n, num_sims = num_sims)

        p1 = 0.5 + es/2
        p2 = 0.5 - es/2
        title_ap1 = "Arm 1 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)
        title_ap2 = "Arm 2 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)

        actions_dir_ts = ts_dir + "bbUnEqualMeansEqualPriorburn_in_size-batch_size={}-{}".format(bs, bs)
        if n == 785:
            hist_wald(df_ts  = df_ts, df_eg0pt1 = df_eg0pt1, df_unif = df_unif, n = n, num_sims = num_sims,title = "With Effect")
            assign_prob_0pt1, ts_ap = hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                         to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                       title = title_ap1, \
                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2", es = es)
                                                  

#        probs_dict = calculate_assgn_prob_by_step_size(actions_root = actions_dir_ts, num_samples=1000, num_actions = 2, cached_probs={},        
#                  prior = [1,1], binary_rewards = True, \
#                  config = {}, n = n, effect_size = es,\
#                  num_sims = num_sims, batch_size = 1, no_effect = False)


 
#        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
#                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
#                                           title = title_ap1, \
#                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")
#        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
#                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
#                                           title = title_ap2, \
#                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_2", mean_key_other = "mean_1")
#        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
#                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts, ts_ap_df = probs_dict,\
#                                           title = title_ap1, \
#                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")
#

        ax[i].set_title("Effect Size = {} \n n = {}".format(es, n_list[i]))
        ax[i].set_ylabel("Power")
        i += 1

        df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
        df.index = ["Uniform Random Wald","Epsilon Greedy Wald", "Thompson Sampling Wald", "Thompson Sampling IPW Wald"]
        df.to_csv("../simulation_analysis_saves/Tables/Power_n={}_numsims={}.csv".format(n, num_sims)) 
        if n == 785:
            summary_table_helper(df_ts = df_ts, df_unif = df_unif,  df_eg0pt1 = df_eg0pt1, effect_size = es, n = n, num_sims = num_sims)


	   
    columns = ['Epsilon Greedy 0.1', "Uniform Random", 'Thompson Sampling', "Thompson Sampling Jefferey's Prior"]
    index = ['Effect Size 0.3', "Effect Size 0.1", 'Effect Size 0.08']
    reward_table_df = pd.DataFrame(reward_table, index = index, columns = columns)
    prop1_table_df = pd.DataFrame(prop1_table, index = index, columns = columns)

    anna_table_save_dir_reward = "../simulation_analysis_saves/Tables/Anna_spec/Reward/"
    Path(anna_table_save_dir_reward).mkdir(parents=True, exist_ok=True)

    anna_table_save_dir_prop1 = "../simulation_analysis_saves/Tables/Anna_spec/Prop1/"
    Path(anna_table_save_dir_prop1).mkdir(parents=True, exist_ok=True)

    save_file = anna_table_save_dir_reward + "reward.csv"
    print("saving to ", save_file)
    reward_table_df.to_csv(save_file)

    save_file = anna_table_save_dir_prop1 + "prop1.csv"
    print("saving to ", save_file)
    prop1_table_df.to_csv(save_file)

    title = "Power Across {} Simulations".format(num_sims)
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






        
root = RL4RL_SECB_DIR + "/simulation_saves/EpsilonGreedyIsEffect"
#parse_dir(root, root_cutoffs)
#num_sims = 500
num_sims = 5000
parse_dir(root, root, num_sims)


