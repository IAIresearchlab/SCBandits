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
#sys.path.insert()
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
from hist_functions import *
import scipy.stats
from pathlib import Path
       # ipdb.set_trace()
import ipdb
from scatter_plot_functions import *
from rectify_vars_and_wald_functions import *
from checkpickle_EFFECT_new import parse_dir

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
        save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)
        Path(save_dir).mkdir(parents=True, exist_ok=True)


        fig_h.savefig(save_dir + "/condition_prop_n={}.png".format(num_steps), bbox_inches = 'tight')
        fig_h.clf()



def stacked_bar_plot_with_cutoff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, df_tsppd = None, n = None, num_sims = None, df_ets = None, \
                     title = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, epsilon = None, es=None):
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = [] #IS REWARD NOT T1 TODO CHnage VAR NAMES
    var_list = []
    t1_list_ts = []
    t1_list_tsppd = []
    t1_list_ets = []
    for num_steps in step_sizes:
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps].dropna()
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps].dropna()
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()
        df_for_num_steps_tsppd = df_tsppd[df_tsppd['num_steps'] == num_steps].dropna()
        df_for_num_steps_ets = df_ets[df_ets['num_steps'] == num_steps].dropna()
        #df_for_num_steps_unif = df_for_num_steps_unif.dropna()
       # bins = np.arange(0, 1.01, .025)


        unif_reward_mean = (df_for_num_steps_unif['total_reward']/num_steps).mean()
        ts_reward_mean = (df_for_num_steps_ts['total_reward']/num_steps).mean()
        eps_greedy_reward_mean_0pt1 = (df_for_num_steps_eg0pt1['total_reward']/num_steps).mean()
        eps_greedy_reward_mean_0pt3 = (df_for_num_steps_eg0pt3['total_reward']/num_steps).mean()

        tsppd_reward_mean = (df_for_num_steps_tsppd['total_reward']/num_steps).mean()
        ets_reward_mean = (df_for_num_steps_ets['total_reward']/num_steps).mean()

        t1_list_unif.append(unif_reward_mean)
        t1_list_ts.append(ts_reward_mean)
        t1_list_eg0pt1.append(eps_greedy_reward_mean_0pt1)
        t1_list_eg0pt3.append(eps_greedy_reward_mean_0pt3)
        t1_list_tsppd.append(tsppd_reward_mean)
        t1_list_ets.append(ets_reward_mean)


       
        
    t1_list_ts = np.array(t1_list_ts)
    t1_list_tsppd = np.array(t1_list_tsppd)
    t1_list_ets = np.array(t1_list_ets)
    ind = np.arange(3*len(step_sizes), step=3)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    print("var", var_list)
    width = 0.44
    capsize = width*4
    width_total = 2*width
    
   
    t1_list_eg0pt1 = np.array(t1_list_eg0pt1)
    t1_list_eg0pt3 = np.array(t1_list_eg0pt3)
    t1_list_unif = np.array(t1_list_unif)
    num_sims_RL4RL = 5000
    t1_eg0pt1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt1*(1-t1_list_eg0pt1)/num_sims_RL4RL) #95 CI for Proportion
    t1_eg0pt3_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt3*(1-t1_list_eg0pt3)/num_sims_RL4RL) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims_RL4RL)
    t1_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts*(1-t1_list_ts)/num_sims_RL4RL)
    num_sims_ppd = num_sims
    t1_se_tsppd = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_tsppd*(1-t1_list_tsppd)/num_sims_ppd)

    num_sims_ets = num_sims
    t1_se_ets = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_ets*(1-t1_list_ets)/num_sims_ets)

    print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
    p1 = ax.bar(ind, t1_list_eg0pt1, width = width, yerr = t1_eg0pt1_se, \
                ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
    p3 = ax.bar(ind+width, t1_list_eg0pt3, width = width, yerr = t1_eg0pt3_se, \
                ecolor='black', capsize=capsize, color = 'green', edgecolor='black')
  
    p4 = ax.bar(ind+2*width, t1_list_ts, width = width, yerr = t1_se_ts,     
               ecolor='black', capsize=capsize, color = 'blue', edgecolor='black') 

    p5 = ax.bar(ind+3*width, t1_list_tsppd, width = width, yerr = t1_se_tsppd,     
               ecolor='black', capsize=capsize, color = 'purple', edgecolor='black') 

    p6 = ax.bar(ind+4*width, t1_list_ets, width = width, yerr = t1_se_ets,     
               ecolor='black', capsize=capsize, color = 'brown', edgecolor='black') 

  
    p2 = ax.bar(ind-width, t1_list_unif, width = width,\
                 yerr = t1_se_unif, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')
    if ax_idx == 2:
    #   leg1 = ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Epsilon Greedy Chi Squared 0.1', "Uniform Chi Squared", "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))
       leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0], p5[0], p6[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald", "PPD c 0.1 Thompson Sampling Wald", "Epsilon 0.1 Thompson Sampling Wald"), bbox_to_anchor=(1.0, 1.76))  
       #leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Chi Squared", 'Epsilon Greedy Chi Squared 0.1', "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))  
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")

    ax.set_ylim(0.40, 0.8)
    
    x = es / 2
    optimal_arm = 0.5 + x
    ax.axhline(y=optimal_arm, linestyle='--')


    return [t1_list_unif, t1_list_eg0pt1, t1_list_ts] #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def parse_dir_old(root, root_cutoffs, num_sims):
    arm_prob= 0.5
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.5, 0.3, 0.1]
    n_list = [32, 88, 785]
    epsilon = 0.1
#EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
    root_dir = root + "/num_sims={}armProb={}".format(num_sims, arm_prob)
    fig, ax = plt.subplots(1,3, figsize = (12,5))
    #fig.set_size_inches(17.5, 13.5)
    ax = ax.ravel()
    i = 0
#    ipdb.set_trace()
    c = 0.1
    num_sims_secb = 5000
    root_ts = "../../../RL4RLSectionB/simulation_saves/IsEffect_fixedbs_RL4RLMay8/num_sims={}armProb=0.5".format(num_sims_secb)
    root_eg = "../../../RL4RLSectionB/simulation_saves/EpsilonGreedyIsEffect/num_sims={}armProb=0.5".format(num_sims_secb)
    root_unif = "../../../RL4RLSectionB/simulation_saves/UniformIsEffect/num_sims={}armProb=0.5".format(num_sims_secb)

    root_ets = "../simulation_saves/EpsilonTSIsEffect/num_sims={}armProb=0.5".format(5000)
    for n in n_list:
        es = es_list[i]
        bs = 1
        es_dir_0pt1 = root_eg + "/es={}epsilon={}/".format(es, 0.1)
        es_dir_0pt3 = root_eg + "/es={}epsilon={}/".format(es, 0.3)
        ts_dir = root_ts + "/es={}/".format(es)
        tsppd_dir = root_dir + "/es={}c={}/".format(es, c)
        ets_dir = root_ets + "/es={}epsilon={}/".format(es, 0.1)
        unif_dir = root_unif + "/es={}/".format(es)

        to_check_eg0pt1 = glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)
      
        to_check_unif = glob.glob(unif_dir + "/*Uniform*{}*{}Df.pkl".format(bs, es))[0]
        assert(len(glob.glob(unif_dir + "/*Uniform*{}*{}Df.pkl".format(bs, es))) == 1)

        to_check_ts = glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_tsppd = glob.glob(tsppd_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(tsppd_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_ets = glob.glob(ets_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!! 
        assert(len(glob.glob(ets_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)


      #------hists, tables etc
        with open(to_check_eg0pt1, 'rb') as f:
            df_eg0pt1 = pickle.load(f)
        with open(to_check_eg0pt3, 'rb') as f:
            df_eg0pt3 = pickle.load(f)
                                               
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ts != None:
            with open(to_check_ts, 'rb') as t:
                df_ts = pickle.load(t)
        with open(to_check_tsppd, 'rb') as f:
            df_tsppd = pickle.load(f)

        with open(to_check_ets, 'rb') as f:
            df_ets = pickle.load(f)
          
#        ipdb.set_trace()
        rect_key = "TS"
        rect_key = "Drop NA"
        rectify_vars_noNa(df_eg0pt1, alg_key = rect_key)
        rectify_vars_noNa(df_eg0pt3, alg_key = rect_key)
        rectify_vars_noNa(df_ts, alg_key = rect_key)
        rectify_vars_noNa(df_unif, alg_key = rect_key)
   
        assert np.sum(df_eg0pt1["wald_type_stat"].isna()) == 0
        assert np.sum(df_eg0pt1["wald_pval"].isna()) == 0
 
        next_df = stacked_bar_plot_with_cutoff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                     df_unif = df_unif, df_ts = df_ts, df_tsppd = df_tsppd, df_ets = df_ets,\
                                             n = n, es=es,num_sims = num_sims,
                                               ax = ax[i], ax_idx = i, epsilon = epsilon)

#

        ax[i].set_title("es = {}, n = {}".format(es, n))
        ax[i].set_ylabel("Reward")
        i += 1

        df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
        df.index = ["Uniform Random Chi Squared","Epsilon Greedy Chi Squared", "Thompson Sampling Chi Squared"]

        save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)

        Path(save_dir).mkdir(parents=True, exist_ok=True)


        df.to_csv(save_dir + "/Reward={}_numsims={}.csv".format(n, num_sims)) 

	   
    title = "Average Reward \n Across {} Simulations".format(num_sims)
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
#    print("saving to ", "plots/{}.png".format(title))
    
    #fig.set_tight_layout(True)
    fig.tight_layout()
    fig.subplots_adjust(top=.8)
    

    fig.savefig(save_dir + "/{}.svg".format(title), bbox_inches = 'tight')
   # plt.show()
    fig.clf()
    plt.close(fig)


if __name__ == "__main__":
    root = "../simulation_saves/TSPPDIsEffect"
    #parse_dir(root, root_cutoffs)
    num_sims = 500
    num_sims = 5000
    title = "Mean Reward \n Averaged Across {} Simulations".format(num_sims)
#    parse_dir_old(root, root, num_sims)
    parse_dir(root, root, num_sims, title = title, metric = "Reward", ylabel = "Reward", ymax = 0.80, num_es = 3)
    title = "Proportion of Optimal Allocations \n Averaged Across {} Simulations".format(num_sims)
    parse_dir(root, root, num_sims, title = title, metric = "PropOpt", ylabel = "Proportion Optimal Allocation", ymax = 1.0, num_es = 3)
