import matplotlib
matplotlib.use('Agg')
import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt 
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
#import explorE_delete as ed
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
import numpy as np
import matplotlib.pyplot as plt
# Set plot font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16


def hist_and_cutoffs(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, plot = True, to_check_ipw_mean1 = None, to_check_ipw_mean2 = None):
    '''
    TODO rename to_check_ipw to to_check_ipw_wald_stat
    '''
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
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
        
        if to_check_ipw != None:
            to_check_ipw_f = to_check_ipw.format(num_steps)
        
            wald_ipw_per_sim = np.load(to_check_ipw_f)
            ipw_mean1 = np.mean(np.load(to_check_ipw_mean1.format(num_steps))) #E[p_hat_mle]
            ipw_mean2 = np.mean(np.load(to_check_ipw_mean2.format(num_steps)))
            
        

        df_unif_for_num_steps = df_unif[df_unif['num_steps'] == num_steps]
        df_unif_for_num_steps_wald = df_unif_for_num_steps['wald_type_stat']
        df_for_num_steps = df[df['num_steps'] == num_steps]
        mle_mean1 = np.mean(df_for_num_steps['mean_1'])
        mle_mean2 = np.mean(df_for_num_steps['mean_2'])
        unif_mean1 = np.mean(df_unif_for_num_steps['mean_1'])
        unif_mean2 = np.mean(df_unif_for_num_steps['mean_2'])
        
        
        df_wald_type_per_sim = df_for_num_steps['wald_type_stat']
      #  df_unif_for_num_steps = np.ma.masked_invalid(df_unif_for_num_steps)
        #print(np.mean(df_unif_for_num_steps))
        if plot == True:
            #ax[i].hist(df_unif_for_num_steps, density = True)
            ax[i].hist(df_unif_for_num_steps_wald, normed = True, alpha = 0.5, \
              label = "Uniform: \n$\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {}".format(
                      np.round(np.mean(df_unif_for_num_steps_wald), 3),\
                      np.round(np.std(df_unif_for_num_steps_wald), 3), np.round(unif_mean1 - 0.5, 3), np.round(unif_mean2 - 0.5, 3)
                      )
              )
            if to_check_ipw != None:

                ax[i].hist(wald_ipw_per_sim, \
                  normed = True, alpha = 0.5,\
                  label = "\n IPW: \n $\mu$ = {} \n$\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {}".format(
                          np.round(np.mean(wald_ipw_per_sim), 3), \
                          np.round(np.std(wald_ipw_per_sim), 3), \
                          np.round(ipw_mean1 - 0.5,3), np.round(ipw_mean2 - 0.5,3)
                          )
                  )
           
            ax[i].hist(df_wald_type_per_sim, \
              normed = True, alpha = 0.5, \
              label = "\n MLE: \n $\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {}".format(
                      np.round(np.mean(df_wald_type_per_sim), 3), \
                      np.round(np.std(df_wald_type_per_sim), 3), \
                      np.round(mle_mean1 - 0.5,3), np.round(mle_mean2 - 0.5,3)
                      )
              )
            ax[i].set_xlabel("number of participants = {} = {}".format(size_vars[i], num_steps))
            ax[i].axvline(x = np.percentile(df_wald_type_per_sim, 2.5), linestyle = "--", color = "black")
            ax[i].axvline(x = np.percentile(df_wald_type_per_sim, 97.5), linestyle = "--", color = "black")
#            ax[i].text(0.85, 0.5,'Mean = {}, Std = {}'.format(np.mean(df_wald_type_per_sim), np.std(df_wald_type_per_sim)),
#             horizontalalignment='center',
#             verticalalignment='center',
#             transform = ax[i].transAxes)
            
         #   ax[i]
            
            
            
            mu = 0
            variance = 1
            sigma = np.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            ax[i].plot(x, stats.norm.pdf(x, mu, sigma))
            ax[i].legend()
            
            #print("mean, std", np.mean(df_wald_type_per_sim), np.std(df_wald_type_per_sim))
              
        
        percenticle_dict_left[str(num_steps)] = np.percentile(df_wald_type_per_sim, 2.5)
        percentile_dict_right[str(num_steps)] = np.percentile(df_wald_type_per_sim, 97.5)
        
        i+=1    
    if plot == True:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        if not os.path.isdir("plots"):
            os.path.mkdir("plots")
        print("saving to ", "plots/{}.png".format(title))
        fig.savefig("plots/{}.png".format(title))

        plt.show()
        plt.clf()
        plt.close()
    
    return percenticle_dict_left, percentile_dict_right


def stacked_bar_plot_with_cutoff(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, percentile_dict_left = None, \
                     percentile_dict_right = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None):
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)

    #print(data)
    
    
    step_sizes = df['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list = []
    t1_wald_list = []
    wald_stat_list = []
    wald_pval_list = []
    arm1_mean_list = []
    arm2_mean_list = []
    
    arm1_std_list = []
    arm2_std_list = []
    ratio_mean_list = []
    ratio_std_list = []
    t1_simbased_list = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []

    for num_steps in step_sizes:
   
        
        df_for_num_steps = df[df['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        

        pval_for_num_steps = df_for_num_steps['pvalue'].mean()
        
        num_replications = len(df_for_num_steps)
 
       # print(num_replications)
   #     if use_pval == True:
        num_rejected = np.sum(df_for_num_steps['pvalue'] < .05)

        num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
        var = np.var(df_for_num_steps_unif['pvalue'] < .05)
        
        t1 =num_rejected / num_replications
        
        t1_unif =num_rejected_unif / num_replications
       
        t1_list_unif.append(t1_unif)
        
        t1_list.append(t1)
        var_list.append(var)
        
    
    ind = np.arange(len(step_sizes))
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    print("var", var_list)
    width = 0.23
    capsize = width*8
    width_total = 2*width
    
   
    t1_list = np.array(t1_list)
    t1_list_unif = np.array(t1_list_unif)
    
    t1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list*(1-t1_list)/num_sims) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims)
    print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
    p1 = ax.bar(ind, t1_list, width = width, yerr = t1_se, \
                ecolor='black', capsize=capsize, color = 'blue', edgecolor='black')
    
    p2 = ax.bar(ind-width, t1_list_unif, width = width,\
                yerr = t1_se_unif, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')
    if ax_idx == 2:
       leg1 = ax.legend((p1[0], p2[0]), ('Epsilon Greedy Chi Squared', "Uniform Chi Squared"), bbox_to_anchor=(1.0, 1.6))
    
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, 0.3)
    ax.axhline(y=0.05, linestyle='--')


    return [t1_list_unif, t1_list] #returns [UR Eps Greedy], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)
   
def get_power_by_steps(dfs_by_trial, columns, alpha = 0.05):
    '''
    df_by_trial is a data frame with information about each run,
    as calculated by calculate_by_trial_statistics_from_sims.
    This function returns the continuous evaluation of power at every step until 4m steps,
    and the power snapshot at step 0.5m, m, 2m and 4m for all effect sizes.
    Power is calculated as what proportion of the p-values were below alpha at that point.
    '''
    unique_sample_sizes = dfs_by_trial[0].num_steps.unique()
    power_df = pd.DataFrame(columns=columns)
    print(power_df)
    power_all_steps = []
    for i in range(len(unique_sample_sizes)):
        
        cur_n = unique_sample_sizes[i]
        print("cur_n", cur_n)
        power_df.loc[i,columns[0]] = cur_n
        # Add bandit and uniform sampling lines 
        j = 1
        for df in dfs_by_trial:
            cur_df = df[df['num_steps'] == cur_n]
            statistic_list = []
            for trial in range(10, cur_n):
                avg_stat = np.sum(cur_df[cur_df['trial'] == trial]['pvalue'] < alpha) \
                            / len(cur_df[cur_df['trial'] == trial])
                statistic_list.append(avg_stat)
            print("len(statistic_list)",len(statistic_list))
            
            if cur_n == unique_sample_sizes[-1]:
                power_all_steps.append(statistic_list[:])
            
           # power_df.iloc[i, j] = statistic_list[-1]
            j+=1
    return power_all_steps, power_df

def plot_trials(t1_list, labels):
    i=0
    fig,ax = plt.subplots()
   # labels= ("Epislon Greedy", "Uniform", "Thompson Sampling")
    colors = ('#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99')
    for p in t1_list:
        x = range(10, len(p)+10)
        ax.plot(x, p, lw=2, color=colors[i], label = labels[i])
        i+=1
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Type 1 Error Rate')
    # set visualization range to 0-1
    ax.set_ylim(0.0, 0.25)
    ax.tick_params(axis='both', which='major')
    plt.title("Type I Error Rate Over Time \n (Binary Reward)")
    plt.legend()
    ax.axhline(0.05, color=colors[0], linestyle='dashed', linewidth=2)

    plt.savefig('plots/TypeIErrorByStep.pdf', bbox_inches='tight')
    


def parse_dir(root):
    num_sims = 500
    arm_prob= 0.5
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.5, 0.3, 0.1]
    n_list = [32, 88, 785]
    epsilon = 0.1
#EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
    root_dir = root + "/num_sims={}armProb={}".format(num_sims, arm_prob)
    fig, ax = plt.subplots(1,3)
    #fig.set_size_inches(17.5, 13.5)
    ax = ax.ravel()
    i = 0
    n = 785

    dir_cur = "../simulation_saves/EpsilonGreedyNoEffect/num_sims={}armProb=0.5/N={}epsilon={}/".format(num_sims, n,epsilon)



    by_trial_eg = dir_cur + "bbUnEqualMeansEqualPriorburn_in_size-batch_size=1-1BB0N{}DfByTrial.pkl".format(n)
    by_trial_unif = dir_cur + "bbUnEqualMeansUniformburn_in_size-batch_size=1-1BU0N785DfByTrial.pkl"

    df_eg = pd.read_pickle(by_trial_eg)
    df_unif = pd.read_pickle(by_trial_unif)

    t1_all_steps_eg, _ = get_power_by_steps([df_eg], ["num_steps", "Epsilon Greedy"], alpha = 0.05)
    t1_all_steps_unif, _ = get_power_by_steps([df_unif], ["num_steps", "Uniform Random"], alpha = 0.05)

     #starts at 10
    labels= ("Epsilon Greedy", "Prior between", "Prior below", "Prior above", "Uniform")

    t1_all_steps_ts = pd.read_pickle("banditsGraphs/180123BinaryPowerSameArms.pkl") #TODO fix paths
    t1_list = [t1_all_steps_eg[0]] + t1_all_steps_ts

    plot_trials(t1_list, labels)



        
root = "EpsilonGreedyNoEffect"
#parse_dir(root, root_cutoffs)
parse_dir(root)


