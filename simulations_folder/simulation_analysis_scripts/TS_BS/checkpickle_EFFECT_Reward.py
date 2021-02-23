import matplotlib
matplotlib.use('Agg')
import pickle
import os
import sys
sys.path.insert(1, '/home/jacobnogas/Sofia_Villar_Fall_2019/banditalgorithms/src/RL4RLSectionB/simulation_analysis_scripts')
import pandas as pd 
from rectify_vars_and_wald_functions import *

import matplotlib.pyplot as plt 
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
       # ipdb.set_trace()
import ipdb


SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 15
reward_file = "../../../empirical_data/experiments_data2.csv"
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('legend', handlelength=SMALL_SIZE)    # legend handle
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def autolabel(rects, ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


def stacked_bar_plot_with_cutoff(df_ts = None, df_unif = None, n = None, num_sims = None, \
                     title = None, \
                     bs_prop = 0.0,\
                     ax = None, ax_idx = None, es=None):
    
    
#    ipdb.set_trace()
    step_sizes = df_ts['num_steps'].unique()
    print(step_sizes)
    size_vars = ["n/2", "n", "2*n", "4*n"]

    unif_reward_list = []
    ts_reward_list = []

    for num_steps in step_sizes:

        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
    
        unif_reward_mean = (df_for_num_steps_unif['total_reward']/num_steps).mean()
        ts_reward_mean = (df_for_num_steps_ts['total_reward']/num_steps).mean()
        unif_reward_list.append(unif_reward_mean)
        ts_reward_list.append(ts_reward_mean)
    
    unif_reward_list = np.array(unif_reward_list)
    ts_reward_list = np.array(ts_reward_list)

    ind = np.arange(len(step_sizes))
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(size_vars)
        
    width = 0.1 
    capsize = width*40 
    width_total = 4*width
    #bs_prop_list =[0.05, 0.10, 0.25]
    if bs_prop == round(1/88, 3):
        bs_prop_prev = 0
        bs_step = 0
    if bs_prop == 0.05:
        bs_prop_prev = round(1/88, 3)
        bs_step = width


    elif bs_prop == 0.10:
        bs_prop_prev = 0.05
        bs_step = 2*width
    elif bs_prop == 0.25:
        bs_prop_prev = 0.10
        bs_step = 3*width
#    else:
 #       bs_prop_prev = 0.10
  #      bs_step = 2*width

    
    unif_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(unif_reward_list*(1-unif_reward_list)/num_sims)
    ts_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(ts_reward_list*(1-ts_reward_list)/num_sims)

    total_width = (0.05 + 0.10 + 0.25)*2*width
    width_prev = width*bs_prop_prev*2 #previous width
    #print(bs_prop, bs_prop_prev)
    width_curr = width*bs_prop*2
   # capsize = width_curr*100
    marker_scaler = 3000
    marker = 's' #filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    
    hatch_dict = {round(1/88,3):"//////", 0.05: "////", 0.10:"//", 0.25: "/"}
    alpha_dict = {round(1/88,3): 0.6, 0.05: 0.7, 0.10: 0.8, 0.25: 0.9}
    hatch_bs = hatch_dict[bs_prop]
    #hatch_bs = None
    alpha_bs = alpha_dict[bs_prop]
    alpha_bs = 0.5
    
#    p1 = ax.bar(ind + width_total + bs_step, t1_simbased_list, width = width, \
#                yerr = t1_simbased_se, ecolor='black', capsize=capsize, \
#                alpha = alpha_bs, color = 'yellow', hatch = hatch_bs, edgecolor='black', label = "Batch size: {} % of sample size".format(100*bs_prop))
   # autolabel(p1, ax)
   
#    p2 = ax.bar(ind-2*width_total + bs_step, ipw_t1_list, width = width, yerr = t1_ipw_se, \
 #               ecolor='green', capsize=capsize, \
  #              alpha = alpha_bs, color = 'green', hatch = hatch_bs, edgecolor='black')
   
    #prev_ind + total_width + width_prev/2 + width_curr
    p3 = ax.bar(ind + bs_step, ts_reward_list, width =width, \
                yerr = ts_se, ecolor='black', \
                capsize=capsize, alpha = alpha_bs, \
                color = 'blue', edgecolor='black', hatch = hatch_bs, label = "Batch size: {} % of sample size".format(round(100*bs_prop, 3)))
   # autolabel(p3, ax)

    prev_ind = ind - 2*width #prev as in t the left
   
    #ind - total_width
    #prev_ind + total_width + width_prev/2 + width_curr \
    if ax_idx == 0:
        p4 = ax.bar(ind - width,\
                unif_reward_list, width = width, \
                yerr = unif_se, ecolor='black',\
                capsize=capsize, \
                color = 'red', edgecolor='black')
    #autolabel(p4, ax)
    #ind - 2*width is start
    #ind - 2*width + width_prev/2 + width_curr
    
#    p5 = ax.bar(ind-2*width_total + bs_step, t1_list_unif, width = width,\
#                yerr = t1_se_unif, ecolor='black', \
#                capsize=capsize, color = 'black', \
#                edgecolor='black', alpha = alpha_bs, hatch = hatch_bs)

    if ax_idx == 0:
        leg1 = ax.legend((p3[0], p4[0]), ('Thompson Sampling \n Wald', "Uniform Wald"), loc = "upper left", bbox_to_anchor=(-0.18, 1.8))
        ax.add_artist(leg1)
    
   # leg2 = ax.legend(loc = 2)
    
    
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants n = {} \n effect size = {}".format(n, es))
    ax.set_ylim(0.40, 0.8)

    x = es / 2
    optimal_arm = 0.5 + x
    ax.axhline(y=optimal_arm, linestyle='--')
    

  


def parse_dir(root, num_sims):
    #num_sims = 1000
    #sims_dir = root + "/num_sims={}".format(num_sims)
    sims_dir = root + "/num_sims={}armProb=0.5/".format(num_sims)
   

    print(os.path.isdir(sims_dir))
    n_list = [88]
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots()
#    fig.set_size_inches(17.5, 13.5)
#    ax = ax.ravel()
    
    i = 0
    bs_prop_list =[1/88, 0.05, 0.10, 0.25] 
    
    for n in n_list:
            es = 0.3
            n_dir = sims_dir + "/es={}/".format(es)
            print(os.path.isdir(n_dir))

#            ipdb.set_trace()
            for bs_prop in bs_prop_list:

                #bs = int(np.floor(bs_prop*n))
                bs = int(np.floor(n*bs_prop))
                bs_prop = round(bs_prop,3)
                print("BS = ", bs, n*bs_prop)
                print("---------------#-------")
                to_check = glob.glob(n_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has uniform and TS, 34 in 348!!
                assert(len(glob.glob(n_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

                to_check_unif = glob.glob(n_dir + "/*Uniform*{}*{}Df.pkl".format(1, es))[0]
                assert(len(glob.glob(n_dir + "/*Uniform*{}*{}Df.pkl".format(1, es))) == 1)

                with open(to_check, 'rb') as f:
                    df_ts = pickle.load(f)
                with open(to_check_unif, 'rb') as f:
                    df_unif = pickle.load(f)

                alg_key = "Drop NA"
                #alg_key = "TS"
               # rectify_vars_noNa(df_eg0pt1, alg_key = alg_key)
               # rectify_vars_noNa(df_eg0pt3, alg_key = alg_key)
                rectify_vars_noNa(df_ts, alg_key = alg_key)
                rectify_vars_noNa(df_unif, alg_key = alg_key)

                title = "Average Reward \n  n = {} and {} sims \n and Batch Size {}".format(n, num_sims, bs)

                stacked_bar_plot_with_cutoff(df_ts = df_ts, df_unif = df_unif,\
					      n = n, num_sims = num_sims, \
					       title = title,\
					       ax = ax, bs_prop = bs_prop, ax_idx = i,es=es \
                             )
		
#            ax[i].set_title("TITLE")
            
                i += 1
    es = 0.3
    x = es / 2
    optimal_arm = 0.5 + x
    other_arm = 0.5-x
    title = "Average Reward Across {} Simulations \n Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(num_sims, es, optimal_arm, other_arm)
        #ax[i].set_title(title, fontsize = 55)
    ax.set_ylabel("Average Reward")
        #i +=1Type 1 Error
        #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	 #   handles, labels = ax[i-1].get_legend_handles_labels()
	  #  fig.legend(handles, labels, loc='upper right', prop={'size': 50})
    #fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right') #prop={'size': 50})
#    fig.suptitle(title, fontsize = 23)
    fig.tight_layout(rect=[0, 0.03, 1, 0.87])
   # fig.tight_layout()
#    fig.subplots_adjust(hspace = 1.0)
    #if not os.path.isdir("plots_multiarm_bs"):
     #   os.mkdir("plots_multiarm_bs")
    print("saving to ", "bs_plots/{}.png".format(title))
#    fig.savefig("../../simulation_analysis_saves/bs_plots/{}.png".format(title), bbox_inches = 'tight')
    save_dir = "../../simulation_analysis_saves/bs_plots"
    if not os.path.isdir(save_dir):
          os.mkdir(save_dir)

    fig.savefig(save_dir + "/{}.png".format(title), bbox_inches = 'tight')

    plt.show()
    plt.clf()
    plt.close()


#REMEMBER TO CHANGE "/" to ":"
num_sims=5000
root_dummy = "../2019-12-06_16:57:17NoEffect"


root = "../../simulation_saves/IsEffect_all_bs_es0pt3"
#Motivational Message ------

parse_dir(root, num_sims)

