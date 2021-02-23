import matplotlib
matplotlib.use('Agg')
import pickle
import os
import sys
sys.path.insert(1, '/home/jacobnogas/Sofia_Villar_Fall_2019/banditalgorithms/src/RL4RLSectionB/simulation_analysis_scripts/')
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
                     ax = None, ax_idx = None):
    
    
#    ipdb.set_trace()
    step_sizes = df_ts['num_steps'].unique()
    print(step_sizes)
    size_vars = ["n/2", "n", "2*n", "4*n"]

    t1_wald_list_ts = []
    t1_wald_list_unif = []

    for num_steps in step_sizes:

        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]

       # SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
       # wald_type_stat = ((mean_1 - mean_2) - delta)/SE #(P^hat_A - P^hat_b)/SE


        num_rejected_wald_ts = np.sum(df_for_num_steps_ts["wald_pval"] < 0.05)
        num_rejected_wald_unif = np.sum(df_for_num_steps_unif["wald_pval"] < 0.05) 

        num_replications = len(df_for_num_steps_ts)
        t1_wald_ts = num_rejected_wald_ts / num_replications

        num_replications = len(df_for_num_steps_unif)
        t1_wald_unif = num_rejected_wald_unif / num_replications
    
        t1_wald_list_ts.append(t1_wald_ts)
        t1_wald_list_unif.append(t1_wald_unif)
    
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

    
    t1_wald_list_ts = np.array(t1_wald_list_ts)
    t1_wald_list_unif = np.array(t1_wald_list_unif)

    t1_wald_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_wald_list_ts*(1-t1_wald_list_ts)/num_sims)
    t1_wald_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_wald_list_unif*(1-t1_wald_list_unif)/num_sims)

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
    p3 = ax.bar(ind + bs_step, t1_wald_list_ts, width =width, \
                yerr = t1_wald_se_ts, ecolor='black', \
                capsize=capsize, alpha = alpha_bs, \
                color = 'blue', edgecolor='black', hatch = hatch_bs, label = "Batch size: {} % of sample size".format(round(100*bs_prop, 3)))
   # autolabel(p3, ax)

    prev_ind = ind - 2*width #prev as in t the left
   
    #ind - total_width
    #prev_ind + total_width + width_prev/2 + width_curr \
    if ax_idx == 0:
        p4 = ax.bar(ind - width,\
                t1_wald_list_unif, width = width, \
                yerr = t1_wald_se_unif, ecolor='black',\
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
        leg1 = ax.legend((p3[0], p4[0]), ('Thompson Sampling \n Wald', "Uniform Wald"), loc = "upper left", bbox_to_anchor=(-0.19, 1.6))
        ax.add_artist(leg1)
    
   # leg2 = ax.legend(loc = 2)
    
    
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants n = {}".format(n))
    ax.set_ylim(0,0.2)
    ax.axhline(y=0.05, linestyle='--')
    
#    plt.tight_layout()
#    if not os.path.isdir("plots"):
#        os.path.mkdir("plots")
#    print("saving in stacked", "plots/{}.png".format(title))
#    plt.savefig("plots/{}.png".format(title))
#    plt.show()
#    plt.clf()
    #ed.add_value_labels(ax)

    label = "BS \n 0.3"


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
            n_dir = sims_dir + "/n={}/".format(n)
            print(os.path.isdir(n_dir))

#            ipdb.set_trace()
            for bs_prop in bs_prop_list:

                #bs = int(np.floor(bs_prop*n))
                bs = int(np.floor(n*bs_prop))
                bs_prop = round(bs_prop,3)
                print("BS = ", bs, n*bs_prop)
                print("---------------#-------")
                to_check = glob.glob(n_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))[0] #Has uniform and TS, 34 in 348!!
                assert(len(glob.glob(n_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))) == 1)

                to_check_unif = glob.glob(n_dir + "/*Uniform*{}*{}Df.pkl".format(1, n))[0]
                assert(len(glob.glob(n_dir + "/*Uniform*{}*{}Df.pkl".format(1, n))) == 1)

                with open(to_check, 'rb') as f:
                    df_ts = pickle.load(f)
                with open(to_check_unif, 'rb') as f:
                    df_unif = pickle.load(f)

                alg_key = "Drop NA"
               # rectify_vars_noNa(df_eg0pt1, alg_key = alg_key)
               # rectify_vars_noNa(df_eg0pt3, alg_key = alg_key)
                rectify_vars_noNa(df_ts, alg_key = alg_key)
                rectify_vars_noNa(df_unif, alg_key = alg_key)

                title = "Type One Error Rates \n  n = {} and {} sims \n and Batch Size {}".format(n, num_sims, bs)

                stacked_bar_plot_with_cutoff(df_ts = df_ts, df_unif = df_unif,\
					      n = n, num_sims = num_sims, \
					       title = title,\
					       ax = ax, bs_prop = bs_prop, ax_idx = i, \
                             )
		
#            ax[i].set_title("TITLE")
            
                i += 1
	   
    title = "False Positive Rates Across {} Simulations".format(num_sims)
        #ax[i].set_title(title, fontsize = 55)
        #i +=1
        #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
    ax.set_ylabel("False Positive Rate")
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	 #   handles, labels = ax[i-1].get_legend_handles_labels()
	  #  fig.legend(handles, labels, loc='upper right', prop={'size': 50})
    #fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right') #prop={'size': 50})
#    fig.suptitle(title, fontsize = 23)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
   # fig.tight_layout()
#    fig.subplots_adjust(hspace = 1.0)
    #if not os.path.isdir("plots_multiarm_bs"):
     #   os.mkdir("plots_multiarm_bs")
    save_dir = "../../simulation_analysis_saves/bs_plots"
    if not os.path.isdir(save_dir):
          os.mkdir(save_dir)

    fig.savefig(save_dir + "/{}.png".format(title), bbox_inches = 'tight')
   # fig.savefig("../../simulation_analysis_saves/bs_plots/{}.png".format(title), bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()











#REMEMBER TO CHANGE "/" to ":"
num_sims=5000

root = "../../simulation_saves/NoEffect_all_bs_n88"


parse_dir(root, num_sims)

