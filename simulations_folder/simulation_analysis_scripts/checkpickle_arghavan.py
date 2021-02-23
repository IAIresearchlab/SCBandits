import matplotlib
matplotlib.use('Agg')
from table_functions import *
import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
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



def plot_hist_and_table(df_for_num_list=[], df_for_num_steps_eg0pt1 = None,
    df_for_num_steps_eg0pt3 = None, df_for_num_steps_ts = None,
    df_for_num_steps_unif = None, num_steps = None, epsilon = None, n = None):

    if len(df_for_num_list):
        result = plot_hist_and_table_freestyle(df_list, num_sims, ax, ax_idx, epsilon)
        return

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


def plot_hist_and_table_freestyle(df_for_num_list=[], num_steps = None, save_dir=None):

    n = df_for_num_list[0]['num_steps'][0]
    fig_h, ax_h = plt.subplots()
    data = []
    for df in df_for_num_list:
        proportions_df_temp = df['sample_size_1'] / num_steps
        ax_h.hist(proportions_df_temp, alpha = 0.5, label = df.name)
        mean_temp = np.mean(proportions_df_temp)
        var_temp = np.var(proportions_df_temp)
        prop_lt_25_temp = np.sum(proportions_df_temp < 0.25) / len(proportions_df_temp)
        data.append([mean_temp, var_temp, prop_lt_25_temp])

    ax_h.legend()
    fig_h.suptitle("Histogram of Proportion of {} Participants Assigned \
        to Condition 1 Across 500 Simulations".format(num_steps))

    #<0.25, 0.25< & <0.5, <0.5 & <0.75, <0.75 & <1.0
    final_data = [['%.3f' % j for j in i] for i in data]
    table = ax_h.table(cellText=final_data,
            colLabels=['Mean', 'Variance', 'prop < 0.25'],
            rowLabels = [df.name for df in df_for_num_list],
            loc='bottom', cellLoc='center', bbox=[0.25, -0.5, 0.5, 0.3])

    if save_dir==None:
        save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)

    Path(save_dir).mkdir(parents=True, exist_ok=True)


    fig_h.savefig(save_dir + "/condition_prop_n={}.png".format(num_steps), bbox_inches = 'tight')
    fig_h.clf()


def stacked_bar_plot_with_cutoff(df_list=[], df_ts = None, df_eg0pt1 = None,
        df_eg0pt3 = None, df_unif = None, n = None, num_sims = None,
        title = None, bs_prop = 0.0, ax = None, ax_idx = None, epsilon = None):

    if len(df_list):
        result = stacked_bar_plot_with_cutoff_freestyle(df_list, ax, ax_idx, epsilon)
        return result

    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []

    for num_steps in step_sizes:
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps].dropna()
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps].dropna()
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()
        #df_for_num_steps_unif = df_for_num_steps_unif.dropna()
        #bins = np.arange(0, 1.01, .025)

        plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon = epsilon, n=n)

 
        #print(num_replications)
        #if use_pval == True:

        #ipdb.set_trace()
        num_replications = len(df_for_num_steps_eg0pt1)
        #num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['pvalue'] < .05) #Epsilon Greedy
        num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['wald_pval'] < .05) #Epsilon Greedy
        #num_rejected_eg0pt1 = np.sum(wald_pval_eg0pt1 < .05) #Epsilon Greedy

        #num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['pvalue'] < .05) #Epsilon Greedy
        num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['wald_pval'] < .05) #Epsilon Greedy

        #num_rejected_ts = np.sum(df_for_num_steps_ts['pvalue'] < .05) #Thompson
        num_rejected_ts = np.sum(df_for_num_steps_ts['wald_pval'] < .05) #Thompson

        #num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
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
        
    t1_list_ts = np.array(t1_list_ts)
    ind = np.arange(3*len(step_sizes), step=3)
    #print(ind)
    #print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    print("var", var_list)
    width = 0.56
    capsize = width*4
    width_total = 2*width

    t1_list_eg0pt1 = np.array(t1_list_eg0pt1)
    t1_list_eg0pt3 = np.array(t1_list_eg0pt3)
    t1_list_unif = np.array(t1_list_unif)
    
    t1_eg0pt1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt1*(1-t1_list_eg0pt1)/num_sims) #95 CI for Proportion
    t1_eg0pt3_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt3*(1-t1_list_eg0pt3)/num_sims) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims)
    t1_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts*(1-t1_list_ts)/num_sims)
    print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
    p1 = ax.bar(ind, t1_list_eg0pt1, width = width, yerr = t1_eg0pt1_se, \
                ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
    p3 = ax.bar(ind+width, t1_list_eg0pt3, width = width, yerr = t1_eg0pt3_se, \
                ecolor='black', capsize=capsize, color = 'green', edgecolor='black')
  
    p4 = ax.bar(ind+2*width, t1_list_ts, width = width, yerr = t1_se_ts,     
               ecolor='black', capsize=capsize, color = 'blue', edgecolor='black') 
  
    p2 = ax.bar(ind-width, t1_list_unif, width = width,\
                 yerr = t1_se_unif, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')
    if ax_idx == 2:
    #   leg1 = ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Epsilon Greedy Chi Squared 0.1', "Uniform Chi Squared", "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))
       leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald"), bbox_to_anchor=(1.0, 1.76))  
       #leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Chi Squared", 'Epsilon Greedy Chi Squared 0.1', "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))  
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
    #plt.tight_layout()
    #plt.title(title)
    #if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, 0.35)
    #ax.set_ylim(0, 0.58)
    ax.axhline(y=0.05, linestyle='--')


    return [t1_list_unif, t1_list_eg0pt1, t1_list_ts] #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)


def stacked_bar_plot_with_cutoff_freestyle(df_list, ax, ax_idx,epsilon):

    num_sims = df_list[0].index[-1]
    step_sizes = df_list[0]['num_steps'].unique()
    size_vars = list(map(str,step_sizes))
    t1_dict = {}
    #t1_se = {}
    var_list = []
    #num_reject_dict = {}
    for df in df_list:
        t1_dict["t1_list_{}".format(df.name)] = []
        #t1_se["t1_se_{}".format(df.name)] = []
        #num_reject_dict["num_reject_{}".format(df.name)] = 0


    for num_steps in step_sizes:
        df_list_temp = []
        for df in df_list:
            df_for_num_steps_temp = df[df['num_steps'] == num_steps].dropna()
            df_for_num_steps_temp.name = df.name + "_for_{}_steps".format(num_steps)
            df_list_temp.append(df_for_num_steps_temp)

            num_rejected = np.sum(df['wald_pval']<0.05)
            num_replications = len(df)
            t1_dict["t1_list_{}".format(df.name)].append(num_rejected/num_replications)

        var = np.var(df_list[0]['pvalue'] < .05)
        var_list.append(var)
        #fix this function
        plot_hist_and_table_freestyle(df_list_temp, num_steps)

    for key in t1_dict:
        t1_dict[key] = np.array(t1_dict[key])

    ind = np.arange(3*len(step_sizes), step=3)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
    print("var", var_list)
    width = 0.56
    capsize = width*4
    width_total = 2*width
    colors = ['yellow', 'green', 'blue', 'red', 'purple']
    clr_idx = 0
    p = []
    for df in df_list:
        t1_temp = t1_dict["t1_list_{}".format(df.name)]
        t1_se_temp = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_temp*(1-t1_temp)/num_sims) #95 CI for Proportion
        #t1_se["t1_se_{}".format(df.name)].append(t1_se_temp)
        p.append(ax.bar(ind, t1_temp, width = width, yerr = t1_se_temp,
            ecolor='black', capsize=capsize, color = colors[clr_idx],
            edgecolor='black'))
        clr_idx += 1

    '''if ax_idx == 2:
        leg1 = ax.legend(p[idx][0] for idx in range(len(df_list))), #(p2[0], p1[0], p3[0], p4[0]),
                ("Uniform Wald",
                "Epsilon Greedy 0.1 Wald",
                "Epsilon Greedy 0.3 Wald",
                "Thompson Sampling Wald"), bbox_to_anchor=(1.0, 1.76))  
        ax.add_artist(leg1)
    '''
    ax.set_xlabel("number of participants = \n {}".format(size_vars))
    ax.set_ylim(0, 0.35)
    #ax.set_ylim(0, 0.58)
    ax.axhline(y=0.05, linestyle='--')

    return list(t1_dict.values()) #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def parse_dir(pickles_list, root, root_cutoffs, num_sims, policy_names = []):
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

    #root_ts = "../simulation_saves/NoEffect_fixedbs_RL4RLMay8/num_sims={}armProb=0.5".format(num_sims)
    df_list = []
    for pk,idx in zip(pickles_list, range(len(pickles_list))):
        #pickle_file = glob.glob(root + pk)[0]
        with open(root + '/' + pk, 'rb') as f:
            df_temp = pickle.load(f)
            df_temp.name = policy_names[idx]

        rect_key = "TS"
        rect_key = "Drop NA"
        rectify_vars_noNa(df_temp, alg_key = rect_key)
        assert np.sum(df_temp["wald_type_stat"].isna()) == 0
        assert np.sum(df_temp["wald_pval"].isna()) == 0
        df_list.append(df_temp)

    n = df_list[0]['num_steps'][0]
    next_df = stacked_bar_plot_with_cutoff(df_list, ax = ax[i], ax_idx = i,
                epsilon = epsilon)
      #  ipdb.set_trace()


    scatter_correlation_helper_outer(df_list)  #Title created in helper fn


    '''
    title_diff = "Difference in Means (|$\hatp_1$ - $\hatp_2$| with MLE) \
        Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = \
        $p_2$ = 0.5".format(n, num_sims)

    hist_means_diff(df_list, title = title_diff)

    title_imba = "Sample Size Imbalance (|$\\frac{n_1}{n} - 0.5$|"+" \
        Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = \
        $p_2$ = 0.5".format(n, num_sims)

    hist_imba(df_list, title = title_imba)
    '''


    ax[i].set_title("n = {}".format(n))
    ax[i].set_ylabel("False Positive Rate")
    i += 1

    step_sizes = df_list[0]['num_steps'].unique()
    size_vars = list(map(str,step_sizes))

    df_t1s = pd.DataFrame(next_df, columns = size_vars)
    #df.index = ["Uniform Random Chi Squared","Epsilon Greedy Chi Squared", "Thompson Sampling Chi Squared"]
    df_t1s.index = [df.name for df in df_list]

    save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)

    Path(save_dir).mkdir(parents=True, exist_ok=True)


    df_t1s.to_csv(save_dir + "/Type1Error_n={}_numsims={}.csv".format(n, num_sims))

    title = "Type 1 Error Rate \n Across {} Simulations".format(num_sims)
    title = "False Positive Rate \n Across {} Simulations".format(num_sims)

    fig.suptitle(title)

    save_dir = "../simulation_analysis_saves/power_t1_plots"
    if not os.path.isdir(save_dir):
          os.mkdir(save_dir)

    fig.tight_layout()
    fig.subplots_adjust(top=.8)


    fig.savefig(save_dir + "/{}.svg".format(title.replace(' ', '').replace('\n', '')),
        bbox_inches = 'tight')
    plt.show()
    fig.clf()
    plt.close(fig)


        
root = "../simulation_saves/arghavan"
pickles_list = ['arghavan=1-1BB0N800Df.pkl']
#parse_dir(root, root_cutoffs)
num_sims = 5000
parse_dir(pickles_list, root, root, num_sims, ['Thompson Sampling'])


