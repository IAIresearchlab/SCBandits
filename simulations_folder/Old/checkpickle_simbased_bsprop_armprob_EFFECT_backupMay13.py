import matplotlib
matplotlib.use('Agg')


import pickle
import os
import pandas as pd 
import ipdb
import matplotlib.pyplot as plt 
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
from hist_functions import *
#import explorE_delete as ed
from rectify_vars_and_wald_functions import *
#figure(num=None, figsize=(15, 15), dpi=60, facecolor='w', edgecolor='k')

#IPW https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
to_check = '2019-08-08_13:19:56/bbUniform0.1BU0.1DfByTrial.pkl'
to_check = 'sim1-2sims/bb0.1BB0.1Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior32BB0N32Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior785BB0N785Df.pkl'
to_check = '2019-08-09_12:49:37-20sims_t1/bbEqualMeansEqualPrior785BB0N785Df.pkl' #10?

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
    
    return percenticle_dict_left, percentile_dict_right

def plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon, es):
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


        table = ax_h.table(cellText=final_data, colLabels=['Mean', 'Variance', 'prop < 0.25'], rowLabels = ["Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"], loc='bottom', cellLoc='center', bbox=[0.25, -0.5, 0.5, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        # Adjust layout to make room for the table:
        #ax_h.tick_params(axis='x', pad=20)

        #fig_h.subplots_adjust(left=0.2, bottom=0.5)
        #fig_h.tight_layout()
        fig_h.savefig("histograms/ExploreAndExploit/es={}/condition_prop_n={}.png".format(es, num_steps), bbox_inches = 'tight')
        fig_h.clf()


def stacked_bar_plot_with_cutoff(df = None, to_check_eg0pt1 = None, to_check_eg0pt3 = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, to_check_ts = None, epsilon = None, es = None):
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
   
#    ipdb.set_trace()
    rectify_vars_noNa(df_eg0pt1, alg_key = "TS")
    rectify_vars_noNa(df_eg0pt3, alg_key = "TS")
    rectify_vars_noNa(df_ts, alg_key = "TS")

 
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

       # bins = np.arange(0, 1.01, .025)

        plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon = epsilon, es=es)

        pval_for_num_steps = df_for_num_steps_eg0pt1['pvalue'].mean()
        num_replications = len(df_for_num_steps_eg0pt1)
 
       # print(num_replications)
   #     if use_pval == True:
        num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['wald_pval'] < .05) #Epsilon Greedy
        num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['wald_pval'] < .05) #Epsilon Greedy
        num_rejected_ts = np.sum(df_for_num_steps_ts['wald_pval'] < .05) #Thompson

        num_rejected_unif = np.sum(df_for_num_steps_unif['wald_pval'] < .05)
        var = np.var(df_for_num_steps_unif['wald_pval'] < .05)
        
        t1_eg0pt1 = num_rejected_eg0pt1 / num_replications
        t1_eg0pt3 = num_rejected_eg0pt3 / num_replications

        t1_ts = num_rejected_ts / num_replications
        t1_unif =num_rejected_unif / num_replications
       
        t1_list_unif.append(t1_unif)
        t1_list_ts.append(t1_ts)
        
        t1_list_eg0pt1.append(t1_eg0pt1)
        t1_list_eg0pt3.append(t1_eg0pt3)
        var_list.append(var)
        
    t1_list_ts = np.array(t1_list_ts)
    ind = np.arange(3*len(step_sizes), step=3)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    print("var", var_list)
    width = 0.60
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
 #      leg1 = ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Epsilon Greedy Chi Squared 0.1', "Uniform Chi Squared", "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))
#       leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Chi Squared", 'Epsilon Greedy Chi Squared 0.1', "Epsilon Greedy Chi Squared 0.3", "Thompson Sampling Chi Squared"), bbox_to_anchor=(1.0, 1.76))  
       leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald"), bbox_to_anchor=(1.0, 1.76))  
 
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, 1.01)
    ax.axhline(y=0.8, linestyle='--')


    return [t1_list_unif, t1_list_eg0pt1, t1_list_ts] #returns [UR Eps_Greedy, TS], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def parse_dir(root, root_cutoffs):
    num_sims = 500
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

    root_ts = "/home/jacobnogas/Sofia_Villar_Fall_2019/banditalgorithms/src/IsEffect_fixedbs_RL4RL/num_sims=500armProb=0.5/"#hard code for now
    for es in es_list:
        #es_dir = root_dir + "/es={}epsilon={}/".format(es, epsilon)
        bs = 1
        n = n_list[i]
        es_dir_0pt1 = root_dir + "/es={}epsilon={}/".format(es, 0.1)
        es_dir_0pt3 = root_dir + "/es={}epsilon={}/".format(es, 0.3)
        ts_dir = root_ts + "/es={}/".format(es)

        to_check_eg0pt1 = glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)
      
        to_check_unif = glob.glob(es_dir_0pt1 + "/*Uniform*{}*{}Df.pkl".format(bs, es))[0]
        assert(len(glob.glob(es_dir_0pt1 + "/*Uniform*{}*{}Df.pkl".format(bs, es))) == 1)

        to_check_ts = glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)
  


        next_df = stacked_bar_plot_with_cutoff(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
                                             n = n_list[i], num_sims = num_sims,
                                               ax = ax[i], ax_idx = i, epsilon = epsilon, es = es)
        title_pval = "Chi Squared P value Disutrbtuion For es = {} \n Across {} Simulations".format(es, num_sims)
        hist_pval(to_check = to_check_eg0pt1, to_check_unif = to_check_unif, to_check_ts = to_check_ts, n = n, num_sims = num_sims, load_df = True, \
                     title = title_pval, plot = True)
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

        title_ap1 = "Arm 1 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)
        title_ap2 = "Arm 2 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n With Effect Size {} \n $p_1$ = {}, $p_2$ = {}".format(n, num_sims, es, p1, p2)

        actions_dir_ts = ts_dir + "bbUnEqualMeansEqualPriorburn_in_size-batch_size={}-{}".format(bs, bs)
                                              
        probs_dict = calculate_assgn_prob_by_step_size(actions_root = actions_dir_ts, num_samples=1000, num_actions = 2, cached_probs={},        
                  prior = [1,1], binary_rewards = True, \
                  config = {}, n = n, effect_size = es,\
                  num_sims = num_sims, batch_size = 1, no_effect = False)


 
#        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
#                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
#                                           title = title_ap1, \
#                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")
#        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
#                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
#                                           title = title_ap2, \
#                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_2", mean_key_other = "mean_1")
        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts, ts_ap_df = probs_dict,\
                                           title = title_ap1, \
                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")

#effect_size = es, no_effect = False

        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts, ts_ap_df = probs_dict,\
                                           title = title_ap2, \
                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_2", mean_key_other = "mean_1")
                                              
        
        ax[i].set_title("Effect Size = {} \n n = {}".format(es, n_list[i]))

        df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
        df.index = ["Uniform Random Chi Squared","Epsilon Greedy Chi Squared", "Thompson Sampling Chi Squared"]
        df.to_csv("Tables/Power_n={}_es={}_numsims={}.csv".format(n_list[i], es,num_sims)) 

        i += 1
	   
    title = "Power Across {} Simulations".format(num_sims)
            #ax[i].set_title(title, fontsize = 55)
            #i +=1
            #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #handles, labels = ax[i-1].get_legend_handles_labels()
    
    #fig.legend(handles, labels, loc='upper right', prop={'size': 50})
        #fig.tight_layout()
    if not os.path.isdir("plots"):
          os.mkdir("plots")
    print("saving to ", "plots/{}.png".format(title))
    fig.tight_layout()
    fig.subplots_adjust(top=.8)

    fig.savefig("plots/{}.svg".format(title), bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()


        
root = "EpsilonGreedyIsEffect"
#parse_dir(root, root_cutoffs)
parse_dir(root, root)


