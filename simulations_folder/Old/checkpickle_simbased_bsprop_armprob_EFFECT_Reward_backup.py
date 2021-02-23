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
                     ax = None, ax_idx = None, df_ts_curr = None, es = None):
    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
        if to_check_ipw != None:
            ipw_t1_list =  np.load(to_check_ipw)

    #print(data)
    
    df_ts_curr_list = np.array(list(df_ts_curr["Prior between"]))
    step_sizes = df['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
 

    unif_reward_list = []
    eps_greedy_reward_list = []

    for num_steps in step_sizes:
   
        n = num_steps
        df_for_num_steps = df[df['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
     

        unif_reward_mean = (df_for_num_steps_unif['total_reward']/n).mean()
        eps_greedy_reward_mean = (df_for_num_steps['total_reward']/n).mean()


        unif_reward_list.append(unif_reward_mean)
        eps_greedy_reward_list.append(eps_greedy_reward_mean)
        
    
    ind = np.arange(2*len(step_sizes), step=2)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
        
    width = 0.5
    capsize = width*5
    width_total = 2*width
    
   
    unif_list = np.array(unif_reward_list)
    eps_list = np.array(eps_greedy_reward_list)
    
    unif_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(unif_list*(1-unif_list)/num_sims) # should be 95 CI for Proportion
    df_ts_curr_list_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(df_ts_curr_list*(1-df_ts_curr_list)/num_sims)
    eps_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(eps_list*(1-eps_list)/num_sims)
    #print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
    p1 = ax.bar(ind, eps_list, width = width, yerr = eps_se, \
                ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
    p2 = ax.bar(ind-width, unif_list, width = width,\
                yerr = unif_se, ecolor='black', \
                capsize=capsize, color = 'red', \
                edgecolor='black')

    p3 = ax.bar(ind+width, df_ts_curr_list, width = width,\
                yerr = df_ts_curr_list_se, ecolor='black', \
                capsize=capsize, color = 'blue', \
                edgecolor='black')


    if ax_idx == 2:
       leg1 = ax.legend((p1[0], p3[0], p2[0]), ('Epsilon Greedy Chi Squared', "Thompson Sampling (Prior Between)","Uniform Chi Squared"), bbox_to_anchor=(1.0, 1.6))
    
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, 0.8)
    
    x = es / 2
    optimal_arm = 0.5 + x
    ax.axhline(y=optimal_arm, linestyle='--')

    return [unif_list, eps_list] #returns [UR Eps Greedy], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)
   
   


def parse_dir(root, root_cutoffs):
    num_sims = 500
    arm_prob= 0.5
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.5, 0.3, 0.1]
    #es_list = [0.5, 0.3] #FOR NOW
    n_list = [32, 88, 785]
    epsilon = 0.1
#EpsilonGreedyIsEffect/num_sims=5armProb=0.5/es=0.3epsilon=0.1/
    root_dir = root + "/num_sims={}armProb={}".format(num_sims, arm_prob)
    fig, ax = plt.subplots(1,3)
    #fig.set_size_inches(17.5, 13.5)
    ax = ax.ravel()
    i = 0
    df_list_ts = pd.read_pickle("banditsGraphs/180114RewardBinary.pkl")

    for es in es_list:
        bs = 1
        es_dir = root_dir + "/es={}epsilon={}/".format(es, epsilon)


        to_check = glob.glob(es_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))[0] #Has uniform and TS, 34 in 348!!
        assert(len(glob.glob(es_dir + "/*Prior*{}*{}Df.pkl".format(bs,es))) == 1)

        to_check_unif = glob.glob(es_dir + "/*Uniform*{}*{}Df.pkl".format(bs, es))[0]
        assert(len(glob.glob(es_dir + "/*Uniform*{}*{}Df.pkl".format(bs, es))) == 1)

     #   to_check_cutoffs = glob.glob(outcome_dir_cutoffs + "/*Prior*{}*{}Df.pkl".format(bs, es))[0] #Has uniform and TS
      #  assert(len(glob.glob(outcome_dir_cutoffs + "/*Prior*{}*{}Df.pkl".format(bs, es))) == 1)

       

        #title = "Power \n  n = {} and {} sims \n Initial Batch Size {} and Batch Size {} \n Arm Prob. {}".format(n, num_sims, bs, bs, arm_prob)
        #percentile_dict_left, percentile_dict_right = hist_and_cutoffs(to_check = to_check_cutoffs, to_check_unif = to_check_unif,\
         #                                                              n = n, num_sims = num_sims, title = title, plot = False) #Note title not used here per say
        df_ts_curr = df_list_ts[2-i]
        next_df = stacked_bar_plot_with_cutoff(to_check = to_check,to_check_unif = to_check_unif,\
                                             n = n_list[i], num_sims = num_sims,
                                               ax = ax[i], ax_idx = i, df_ts_curr = df_ts_curr, es=es)


        df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
        df.index = ["Uniform Random","Epsilon Greedy"]
        df.to_csv("Tables/Reward_n={}_es={}_numsims={}.csv".format(n_list[i], es,num_sims)) 

                
        ax[i].set_title("Effect Size = {} \n n = {}".format(es, n_list[i]))
        i += 1
	   
    title = "Reward Across {} Simulations For Epsilon = {}".format(num_sims, epsilon)
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


