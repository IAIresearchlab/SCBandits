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

from get_assistments_rewards import *


#import explorE_delete as ed
#figure(num=None, figsize=(15, 15), dpi=60, facecolor='w', edgecolor='k')

#IPW https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
to_check = '2019-08-08_13:19:56/bbUniform0.1BU0.1DfByTrial.pkl'
to_check = 'sim1-2sims/bb0.1BB0.1Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior32BB0N32Df.pkl'
to_check = '2019-08-09_12:39:47/bbEqualMeansEqualPrior785BB0N785Df.pkl'
to_check = '2019-08-09_12:49:37-20sims_t1/bbEqualMeansEqualPrior785BB0N785Df.pkl' #10?

MALL_SIZE = 8
MEDIUM_SIZE = 17
BIGGER_SIZE = 20
reward_file = "../../../empirical_data/experiments_data2.csv"
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
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


def hist_and_cutoffs(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, plot = True, to_check_ipw_mean1 = None, to_check_ipw_mean2 = None, \
                      reward_header = None, condition_header = None, center = None, delta_nonzero = False, title_save = None):
    '''
    TODO rename to_check_ipw to to_check_ipw_wald_stat
    delta_nonzero should be false for cutoffs, since sim w.o delta
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
    percentile_dict_left = {}
    percentile_dict_right = {}

    percentile_dict_left_ipw = {}
    percentile_dict_right_ipw = {}



    for num_steps in step_sizes:
        
        if to_check_ipw != None:
            to_check_ipw_f = to_check_ipw.format(num_steps)
        
            wald_ipw_per_sim = np.load(to_check_ipw_f)
            if to_check_ipw_mean1 != None:
                ipw_mean1 = np.mean(np.load(to_check_ipw_mean1.format(num_steps))) #E[p_hat_mle]
                ipw_mean2 = np.mean(np.load(to_check_ipw_mean2.format(num_steps)))
                ipw_var1 = np.var(np.load(to_check_ipw_mean1.format(num_steps)))
                ipw_var2 = np.var(np.load(to_check_ipw_mean2.format(num_steps)))
        
        df_unif_for_num_steps = df_unif[df_unif['num_steps'] == num_steps]
        df_unif_for_num_steps_wald = df_unif_for_num_steps['wald_type_stat']
        df_for_num_steps = df[df['num_steps'] == num_steps]
        #---------------------TS delta
        mean_1 = df_for_num_steps["mean_1"]
        mean_2 = df_for_num_steps["mean_2"]
        sample_size_1 = df_for_num_steps["sample_size_1"]
        sample_size_2 = df_for_num_steps["sample_size_2"]

        
    
        arm_1_rewards, arm_2_rewards = read_pcrs_rewards(reward_file, reward_header, condition_header) #true rewards

        p1 = sum(arm_1_rewards)/len(arm_1_rewards)
        p2 = sum(arm_2_rewards)/len(arm_2_rewards)
        print(p1, p2)
        if delta_nonzero == True:
            delta = p1 - p2
        else:
            delta = 0
        
        #delta=0
        SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
        wald_type_stat = ((mean_1 - mean_2) - delta)/SE #(P^hat_A - P^hat_b)/SE #Don't subtract delta when simulated for cutoffs delta=0

        #--------------------TS delta

        #---------------------IPW delta ---Assuming not IPW from cutoffs ><
        mean_1 = np.load(to_check_ipw_mean1.format(num_steps))
        mean_2 = np.load(to_check_ipw_mean2.format(num_steps))
        
        #delta=0
        SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
        wald_type_stat_ipw_delta = ((mean_1 - mean_2) - delta)/SE #(P^hat_A - P^hat_b)/SE #Don't subtract delta when simulated for cutoffs delta=0

        #--------------------IPW delta



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
        unif_var1 = np.var(df_unif_for_num_steps['mean_1'])
        unif_var2 = np.var(df_unif_for_num_steps['mean_2'])
        
        
        df_wald_type_per_sim = df_for_num_steps['wald_type_stat']
      #  df_unif_for_num_steps = np.ma.masked_invalid(df_unif_for_num_steps)
        #print(np.mean(df_unif_for_num_steps))
       
        if plot == True:
            #ax[i].hist(df_unif_for_num_steps, density = True)
#changed from Uniform for debugging
            ax.hist(wald_type_stat_unif, normed = True, alpha = 0.5, \
              label = "Uniform: \n$\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} \n var($\hatp_2$) = {}".format(
                      np.round(np.mean(wald_type_stat_unif), 3),\
                      np.round(np.std(wald_type_stat_unif), 3), np.round(unif_mean1 - p1, 3), np.round(unif_mean2 - p2, 3), np.round(unif_var1, 3), np.round(unif_var2, 3)
                      )
              )
            if to_check_ipw != None:

                ax.hist(wald_type_stat_ipw_delta, \
                  normed = True, alpha = 0.5,\
                  label = "\n TS IPW: \n $\mu$ = {} \n$\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} \n var($\hatp_2$) = {}".format(
                          np.round(np.mean(wald_type_stat_ipw_delta), 3), \
                          np.round(np.std(wald_type_stat_ipw_delta), 3), \
                          #np.round(ipw_mean1 - true_prob_per_arm[0],3), np.round(ipw_mean2 - true_prob_per_arm[1],3),\
                          np.round(ipw_mean1 - p1,3), np.round(ipw_mean2 - p2,3), np.round(ipw_var1, 3), np.round(ipw_var2, 3)
                          )
                  )
           
            ax.hist(wald_type_stat, \
              normed = True, alpha = 0.5, \
              label = "\n TS MLE: \n $\mu$ = {} \n $\sigma$ = {} \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {} \n var($\hatp_1$) = {} \n var($\hatp_2$) = {}".format(
                      np.round(np.mean(wald_type_stat), 3), \
                      np.round(np.std(wald_type_stat), 3), \
                      np.round(mle_mean1 - p1,3), np.round(mle_mean2 - p2, 3), np.round(mle_var1, 3), np.round(mle_var2, 3)
                      )
              )
           # ax.set_xlabel("number of participants = {} = {}".format(size_vars[i], num_steps))
           # ax.axvline(x = np.percentile(wald_type_stat, 2.5), linestyle = "--", color = "black")
           # ax.axvline(x = np.percentile(wald_type_stat, 97.5), linestyle = "--", color = "black")

          #  ax.text(np.percentile(wald_type_stat, 2.5), 0.35, np.round(np.percentile(wald_type_stat, 2.5), 3))
           # ax.text(np.percentile(wald_type_stat, 97.5), 0.35, np.round(np.percentile(wald_type_stat, 97.5),3))

           # ax.axvline(x = np.percentile(wald_type_stat_unif, 2.5), linestyle = "--", color = "red")
           # ax.axvline(x = np.percentile(wald_type_stat_unif, 97.5), linestyle = "--", color = "red")
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
              
        
        percentile_dict_left[str(num_steps)] = np.percentile(wald_type_stat, 2.5)
        percentile_dict_right[str(num_steps)] = np.percentile(wald_type_stat, 97.5)

        percentile_dict_left_ipw[str(num_steps)] = np.percentile(wald_ipw_per_sim, 2.5)
        percentile_dict_right_ipw[str(num_steps)] = np.percentile(wald_ipw_per_sim, 97.5)
        
        i+=1    
    if plot == True:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.80])
        #if not os.path.isdir("Histograms"):
         #   os.path.mkdir("plots")
        print("saving to ", "Centers_varying/Histograms/{}.png".format(title_save))
        fig.savefig("Centers_varying/Histograms/{}.png".format(title_save))

        plt.show()
        plt.clf()
        plt.close()
    
    cell_str_pcrs = "TS PCRS: \n p1 = {}, p2 = p{}, \n 2.5% percentile = {} , 97.5 % percetile = {} \n $\hat\mu = {}, \hat\sigma = {}$ \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {}".format(np.round(p1,3), np.round(p2,3), \
       np.round(np.percentile(wald_type_stat_unif, 2.5),3), np.round(np.percentile(wald_type_stat_unif, 97.5),3), \
        np.round(np.mean(wald_type_stat_unif),3), np.round(np.std(wald_type_stat_unif),3), np.round(unif_mean1 - p1, 3), np.round(unif_mean2 - p2, 3))

    cell_str_synthco = "TS Synthetic for Cutoffs: \n p1 = {}, p2 = p{}, \n 2.5% percentile = {} , 97.5 % percetile = {} \n $\hat\mu = {}, \hat\sigma = {}$ \n bias($\hatp_1$) = {} \n bias($\hatp_2$) = {}".format(center, center, \
       np.round(np.percentile(wald_type_stat, 2.5), 3), np.round(np.percentile(wald_type_stat, 97.5),3), \
        np.round(np.mean(wald_type_stat),3), np.round(np.std(wald_type_stat),3), np.round(mle_mean1 - center, 3), np.round(mle_mean1 - center, 3))

    cell_str_pcrs = "TS PCRS: \n 2.5% percentile = {} , 97.5 % percentile = {}".format(np.round(np.percentile(wald_type_stat_unif, 2.5),3), np.round(np.percentile(wald_type_stat_unif, 97.5),3))

    cell_str_synthco = "TS Synthetic for Cutoffs: \n 2.5% percentile = {} , 97.5 % percentile = {}".format(
       np.round(np.percentile(wald_type_stat, 2.5), 3), np.round(np.percentile(wald_type_stat, 97.5),3))

    cell_str = cell_str_pcrs + "\n \n" + cell_str_synthco

    return percentile_dict_left, percentile_dict_right, percentile_dict_left_ipw, percentile_dict_right_ipw, cell_str


def stacked_bar_plot_with_cutoff(df = None, to_check = None, to_check_unif = None, to_check_ipw = None, n = None, num_sims = None, load_df = True, \
                     title = None, percentile_dict_left = None, \
                     percentile_dict_right = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, outcomes_list = None,\
 reward_header = None, condition_header = None, to_check_ipw_mean1 = None,\
 to_check_ipw_mean2 = None, percentile_dict_left_ipw = None, percentile_dict_right_ipw = None):

    if load_df == True:
        with open(to_check, 'rb') as f:
            df = pickle.load(f)
        with open(to_check_unif, 'rb') as f:
            df_unif = pickle.load(f)
   #     if to_check_ipw != None:
   #         ipw_t1_list =  np.load(to_check_ipw)

    #print(data)
    
    
    
    step_sizes = df['num_steps'].unique()
    print(step_sizes)
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
    t1_simbased_list_ipw = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    ipw_t1_list = []

    #for num_steps in step_sizes:
    #    by_sim = df.groupby("num_steps").mean()
    #    by_sim.index.names = ['num_steps']



    num_steps = n    
    df_for_num_steps = df[df['num_steps'] == num_steps]
    df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]

    #---------------------TS delta
    mean_1 = df_for_num_steps["mean_1"]
    mean_2 = df_for_num_steps["mean_2"]
    sample_size_1 = df_for_num_steps["sample_size_1"]
    sample_size_2 = df_for_num_steps["sample_size_2"]

    
    
    arm_1_rewards, arm_2_rewards = read_pcrs_rewards(reward_file, reward_header, condition_header) #true rewards

    p1 = sum(arm_1_rewards)/len(arm_1_rewards)
    p2 = sum(arm_2_rewards)/len(arm_2_rewards)
    print(p1, p2)
    delta = p1 - p2
    #delta = 0
    SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
    wald_type_stat = ((mean_1 - mean_2))/SE #(P^hat_A - P^hat_b)/SE

    #--------------------TS delta


    #---------------------TS IPW
    mean_1 = np.load(to_check_ipw_mean1) #E[p_hat_mle]
    mean_2 = np.load(to_check_ipw_mean2)
    sample_size_1 = df_for_num_steps["sample_size_1"]
    sample_size_2 = df_for_num_steps["sample_size_2"]

    
    
    arm_1_rewards, arm_2_rewards = read_pcrs_rewards(reward_file, reward_header, condition_header) #true rewards

    p1 = sum(arm_1_rewards)/len(arm_1_rewards)
    p2 = sum(arm_2_rewards)/len(arm_2_rewards)
    print(p1, p2)
    delta = p1 - p2
    #delta = 0
    SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
    wald_type_stat_ipw = ((mean_1 - mean_2))/SE #(P^hat_A - P^hat_b)/SE

    #--------------------TS IPW

    #---------------------unif delta
    mean_1 = df_for_num_steps_unif["mean_1"]
    mean_2 = df_for_num_steps_unif["mean_2"]
    sample_size_1 = df_for_num_steps_unif["sample_size_1"]
    sample_size_2 = df_for_num_steps_unif["sample_size_2"]

    SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) 
    wald_type_stat_unif = ((mean_1 - mean_2))/SE #(P^hat_A - P^hat_b)/SE

    #--------------------unif delta



    
    #print("------{}-------".format(title))
    
   # print(wald_type_stat)
    #print("mean_1","mean_2", mean_1,mean_2)
    #print("sample_size_1","sample_size_2", sample_size_1,sample_size_2)
    #print("wald_type_stat", wald_type_stat)
    #print("SE", SE)
    #print("------")
    
#   mean1 = np.mean(df_for_num_steps['mean_1'])
#   mean1 = np.mean(df_for_num_steps['mean_2'])
    
    wald_pval_for_num_steps = df_for_num_steps['wald_pval'].mean()
    wald_stat_for_num_steps = df_for_num_steps['wald_type_stat'].mean()   
    
    left_cutoff = percentile_dict_left[str(num_steps)]
    right_cutoff = percentile_dict_right[str(num_steps)]

    left_cutoff_ipw = percentile_dict_left_ipw[str(num_steps)]
    right_cutoff_ipw = percentile_dict_right_ipw[str(num_steps)]
 
    
    wald_pval_list.append(wald_pval_for_num_steps)        
    wald_stat_list.append(wald_stat_for_num_steps) #UNUSED

    num_rejected_wald = np.sum(np.abs(wald_type_stat) > stats.norm.ppf(0.975))
    num_rejected_wald_unif = np.sum(np.abs(wald_type_stat_unif) > stats.norm.ppf(0.975))
    num_rejected_wald_ipw = np.sum(np.abs(wald_type_stat_ipw) > stats.norm.ppf(0.975))

    

   


 #   pval_for_num_steps = df_for_num_steps['pvalue'].mean()
    
    num_replications = len(df_for_num_steps)
   #if use_pval == True:
    num_rejected = np.sum(df_for_num_steps['pvalue'] < .05)
    
   # num_rejected_wald = np.sum(df_for_num_steps['wald_pval'] < .05)

    num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
   # num_rejected_wald_unif = np.sum(df_for_num_steps_unif['wald_pval'] < .05)
   # num_rejected_wald_unif = np.sum(df_for_num_steps_unif['wald_pval'] < .05)
    
   # rejected_wald_simbased = np.sum(np.logical_or(df_for_num_steps['wald_type_stat'] > right_cutoff,\
    #          df_for_num_steps['wald_type_stat'] < left_cutoff)) #two sided test with empirical cut offs

    rejected_wald_simbased = np.sum(np.logical_or(wald_type_stat > right_cutoff,\
              wald_type_stat < left_cutoff)) #two sided test with empirical cut offs

    rejected_wald_simbased_ipw = np.sum(np.logical_or(wald_type_stat_ipw > right_cutoff_ipw,\
              wald_type_stat_ipw < left_cutoff_ipw)) #two sided test with empirical cut offs  
  
    #  print(num_rejected_wald_simbased)
    # print(num_rejected_wald, num_rejected_wald_ppf)
    
    t1_simbased = rejected_wald_simbased/num_replications
    t1_simbased_ipw = rejected_wald_simbased_ipw/num_replications
    
    t1 =num_rejected / num_replications
    t1_wald = num_rejected_wald / num_replications

    t1_ipw = num_rejected_wald_ipw / num_replications   

    t1_unif =num_rejected_unif / num_replications
    t1_wald_unif = num_rejected_wald_unif / num_replications
    
    t1_list_unif.append(t1_unif)
    t1_wald_list_unif.append(t1_wald_unif)
    
    t1_list.append(t1)
    t1_wald_list.append(t1_wald)
    t1_simbased_list.append(t1_simbased)
    t1_simbased_list_ipw.append(t1_simbased_ipw)
    ipw_t1_list.append(t1_ipw)
#   
#        arm1_mean_list.append(arm1_mean)
#        arm2_mean_list.append(arm2_mean)
#        arm1_std_list.append(arm1_std)
#        arm2_std_list.append(arm2_std)
#        
#        ratio_mean_list.append(ratio_mean)
#        ratio_std_list.append(ratio_std)
    
#    plt.suptitle(title)
#    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#    plt.show()
    
    ind = np.arange(len([n]))
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels([""])
        
    width = 0.055 
    capsize = width*100 
    width_total = 3*width
    
    #markers, stemlines, baseline = ax.stem(ind + 2*width, t1_simbased_list)
    #ax.set(markers, markersize= bs_prop*100)
    
    #bs_prop_list =[0.05, 0.10, 0.25]
    if bs_prop == 0.05:
        bs_prop_prev = 0
        bs_step = 0
    elif bs_prop == 0.10:
        bs_prop_prev = 0.05
        bs_step = width
    else:
        bs_prop_prev = 0.10
        bs_step = 2*width
    #ax.axvline(ind+2*width, t1_simbased_list)
   # bs_prop = np.log(bs_prop + 1.1)
   # bs_prop_prev = np.log(bs_prop_prev + 1.1)
    
    t1_simbased_list = np.array(t1_simbased_list)
    t1_simbased_list_ipw = np.array(t1_simbased_list_ipw)
    t1_list = np.array(t1_list)
    t1_wald_list = np.array(t1_wald_list)
    t1_wald_list_unif = np.array(t1_wald_list_unif)
    t1_list_unif = np.array(t1_list_unif)
    ipw_t1_list = np.array(ipw_t1_list)    

    t1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list*(1-t1_list)/num_sims) #95 CI for Proportion
    t1_wald_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_wald_list*(1-t1_wald_list)/num_sims)
    t1_ipw_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(ipw_t1_list*(1-ipw_t1_list)/num_sims)
    t1_simbased_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_simbased_list*(1-t1_simbased_list)/num_sims)

    t1_simbased_ipw_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_simbased_list_ipw*(1-t1_simbased_list_ipw)/num_sims)
   
    
    #print(t1_simbased_list, t1_list, t1_wald_list, t1_wald_list_unif, t1_list_unif)
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims)
    
    t1_wald_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_wald_list_unif*(1-t1_wald_list_unif)/num_sims)
    total_width = (0.05 + 0.10 + 0.25)*2*width
    width_prev = width*bs_prop_prev*2 #previous width
    #print(bs_prop, bs_prop_prev)
    width_curr = width*bs_prop*2
   # capsize = width_curr*100
    marker_scaler = 3000
    marker = 's' #filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    
    hatch_dict = {0.05: "////", 0.10:"//", 0.25: "/"}
    alpha_dict = {0.05: 0.7, 0.10: 0.8, 0.25: 0.9}
    hatch_bs = hatch_dict[bs_prop]
    #hatch_bs = None
    alpha_bs = alpha_dict[bs_prop]
    alpha_bs = 0.5
   # ax.scatter(ind+2*width, t1_simbased_list, bs_prop*marker_scaler, color = 'blue', alpha = 0.7, \
    #           label = "Batch Size {}".format(bs_prop), marker = marker)
#ind + width_total + bs_step
    
    p1 = ax.bar(ind + bs_step, t1_simbased_list, width = width, \
                yerr = t1_simbased_se, ecolor='black', capsize=capsize, \
                alpha = alpha_bs, color = 'yellow', hatch = hatch_bs, edgecolor='black', label = "Batch size: {} % of sample size".format(100*bs_prop))
   # autolabel(p1, ax)
    #ax.scatter(ind + width, t1_list, bs_prop*marker_scaler, color = 'green', \
     #          alpha = 0.7,  marker = marker) #label = 'Thompson Chi Squared BS {}'.format(bs_prop))
   
    p2 = ax.bar(ind-width_total + bs_step, ipw_t1_list, width = width, yerr = t1_ipw_se, \
                ecolor='green', capsize=capsize, \
                alpha = alpha_bs, color = 'green', hatch = hatch_bs, edgecolor='black')
   
   # ax.scatter(ind, t1_wald_list, bs_prop*marker_scaler, color = 'red', \
              # alpha = 0.7, marker = marker)# label = 'Thompson Wald BS {}'.format(bs_prop))
    print("t1 = {} for bs {}".format(t1_wald_list, bs_prop))
    #prev_ind + total_width + width_prev/2 + width_curr
    p3 = ax.bar(ind - 2*width_total+ bs_step, t1_wald_list, width =width, \
                yerr = t1_wald_se, ecolor='black', \
                capsize=capsize, alpha = alpha_bs, \
                color = 'blue', edgecolor='black', hatch = hatch_bs)
   # autolabel(p3, ax)
   # ax.scatter(ind - width, t1_wald_list_unif, \
            #   bs_prop*marker_scaler, color = 'brown', \
           #    alpha = 0.7,  marker = marker) #label = "Uniform Wald BS {}".format(bs_prop))

    prev_ind = ind - 2*width #prev as in t the left
   
    #ind - total_width
    #prev_ind + total_width + width_prev/2 + width_curr \
#    p4 = ax.bar(ind - 2*width_total + bs_step
#                , t1_wald_list_unif, width = width, \
#                yerr = t1_wald_se_unif, ecolor='black',\
#                capsize=capsize, alpha = alpha_bs, \

#                color = 'red', edgecolor='black', hatch = hatch_bs)
    #autolabel(p4, ax)
   # ax.scatter(ind - 2*width, t1_list_unif, bs_prop*marker_scaler, \
    #           color = 'black', alpha = 0.7, marker= marker)# label = "Uniform Chi Squared BS {}".format(bs_prop))

    #ind - 2*width is start
    #ind - 2*width + width_prev/2 + width_curr
    
#    p5 = ax.bar(ind+2*width_total + bs_step, t1_simbased_list_ipw, width = width,\
#                yerr = t1_simbased_ipw_se, ecolor='black', \
#                capsize=capsize, color = 'black', \
#                edgecolor='black', alpha = alpha_bs, hatch = hatch_bs)

    if ax_idx == 0:
#       leg1 = ax.legend((p5[0], p1[0], p2[0], p3[0], p4[0]), ("Empirical Cutoffs IPW", "Empirical Cutoffs TS", "IPW",\
#                   'TS', "Uniform"), loc = "upper left", bbox_to_anchor=(-0.01, 2.3))
       leg1 = ax.legend((p3[0], p2[0], p1[0]), ("Wald Test Statistic",'IPW-Adjustment to Wald Test Statistic',  "Algorithm-Induced Wald Test Statistic"), loc = "upper left", bbox_to_anchor=(-0.01, 1.3))

       ax.add_artist(leg1)
    
   # leg2 = ax.legend(loc = 2)
    
    
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    #ax.set_xlabel("number of participants = {}".format(n))
    if bs_prop == 0.25:
        lim_unif = t1_wald_list_unif[0]
        print("lim_unif", lim_unif)
        ax.axhline(y=lim_unif, linestyle='--')
    ax.set_ylim(0, 1.01)
 
    return [t1_wald_list_unif[0], t1_wald_list[0], t1_simbased_list[0], ipw_t1_list[0], t1_simbased_list_ipw[0]] #returns [UR TS TSEC IPW], also en  lists so take [0]




def parse_dir(root, root_cutoffs, num_sims, outcomes_title_dict, sample_size_dict, experiment_title, condition_header):
    #num_sims = 1000
    #sims_dir = root + "/num_sims={}".format(num_sims)
    sims_dir = root.format(num_sims)
   

    print(os.path.isdir(sims_dir))
    sims_dir_cutoffs = root_cutoffs.format(num_sims)
    #n = 302
    #../../Outfile/ParamsBased/numSims5000motivate_final//Y1/bbEqualMeansEqualPriorburn_in_size-batch_size=17-17BB0N348Df.pkl#
    #n_list = [302]

    outcomes_list = ["Y1", "Y2", "Y3"] #348 640 460
    outcomes_list = ["Y4"] #348 640 460
    prob_success_dict = {"Y1":[0.5, 0.75], "Y2":[0.5, 0.75], "Y3":[0.5, 0.75], "Y4":[0.25,0.5,0.75]}

    i = 0
    bs_prop_list =[0.05, 0.10, 0.25]
    
    
    for outcome in outcomes_list: #one plot per outcome, and also one table
            n=sample_size_dict[outcome]
            fig, ax = plt.subplots(3,1)
            fig, ax = plt.subplots()
            #fig, ax = plt.subplots()
            fig.set_size_inches(17.5, 13.5)
#            ax = ax.ravel()

            outcome_dir = sims_dir + "/{}/".format(outcome)
            outcome_dir_cutoffs = sims_dir_cutoffs + "/{}/".format(outcome)
            i = 0#for subplots
            table_out = []
            df = pd.DataFrame()
            output_str = "Sample Size: {} \n".format(n)
        
            #for center in ["0pt25","0pt5","0pt75"]:
            for center in ["0pt5"]:
                output_str += "TS Cutoffs with Synthetic Means p0 = p1 = {} \n".format(center.replace("pt", "."))
           
                outcome_dir_cutoffs_center = outcome_dir_cutoffs + "/"+ "center{}".format(center) + "/"
                if not os.path.isdir(outcome_dir_cutoffs_center):
                    print(outcome_dir_cutoffs_center, "does not exist")

                else:
                    print(outcome_dir_cutoffs_center, "does exist")

                control_rewards, int_rewards = read_pcrs_rewards_control_as_arm1(reward_file, outcome, condition_header) 
                

                p0 = sum(control_rewards)/len(control_rewards)
                p1 = sum(int_rewards)/len(int_rewards)
                delta = np.round(np.abs(p0-p1), 3)
                p0 = np.round(sum(control_rewards)/len(control_rewards), 3)
                p1 = np.round(sum(int_rewards)/len(int_rewards), 3)
                
                output_str += "TS With PCRS Data Means p0 = {}, p1 = {}".format(p0,p1)
                next_table_dict = {}
                col = []
                for bs_prop in bs_prop_list:
    
                    #bs = int(np.floor(bs_prop*n))
                    
                    bs = int(np.floor(n*bs_prop))
                    print("BS = ", bs, n*bs_prop)
                    print("---------------#-------")
                    to_check = glob.glob(outcome_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))[0] #Has uniform and TS, 34 in 348!!
                    assert(len(glob.glob(outcome_dir + "/*Prior*{}*{}Df.pkl".format(bs,n))) == 1)
    
                    to_check_unif = glob.glob(outcome_dir + "/*Uniform*{}*{}Df.pkl".format(bs, n))[0]
                    assert(len(glob.glob(outcome_dir + "/*Uniform*{}*{}Df.pkl".format(bs, n))) == 1)
    
                    to_check_cutoffs = glob.glob(outcome_dir_cutoffs_center + "/*Prior*{}*{}Df.pkl".format(bs, n))[0] #Has uniform and TS
                    assert(len(glob.glob(outcome_dir_cutoffs_center + "/*Prior*{}*{}Df.pkl".format(bs, n))) == 1)
    
                    ipw_t1 = outcome_dir + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_t1_per_num_participants.npy".format(bs,bs,n)
                    to_check_ipw_mean1 = outcome_dir + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_mean_1_{}_participants.npy".format(bs,bs,n)
                    to_check_ipw_mean2 = outcome_dir + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_mean_2_{}_participants.npy".format(bs,bs,n)
                    to_check_ipw = outcome_dir + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_wald_stat_per_sim_{}.npy".format(bs,bs,n)


                    to_check_ipw_mean1_cutoffs = outcome_dir_cutoffs_center + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_mean_1_{}_participants.npy".format(bs,bs,n)
                    to_check_ipw_mean2_cutoffs = outcome_dir_cutoffs_center + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_mean_2_{}_participants.npy".format(bs,bs,n)
                    to_check_ipw_cutoffs = outcome_dir_cutoffs_center + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}-ipw_wald_stat_per_sim_{}.npy".format(bs,bs,n)

                    title = "Histogram of Wald-z Statistic \n Across {} Simulations \n".format(num_sims)

                    title_save = "Cutoffs From p1 = p2 = {} Experiment: {} Outcome: {} n = {} and {} sims Batch Size {} p_0 = {}, p_1 = {}, |p_0 - p_1| = {}".format(float(center.replace("pt",".")), experiment_title, outcomes_title_dict[outcome], n, num_sims, bs, p0, p1, delta)
                    title_save = title_save.replace(".","pt")

                    percentile_dict_left, percentile_dict_right, \
                    percentile_dict_left_ipw, percentile_dict_right_ipw, \
                     cell_str = hist_and_cutoffs(to_check = to_check_cutoffs, to_check_unif = to_check_unif, to_check_ipw = to_check_ipw_cutoffs,\
    								       n = n, num_sims = num_sims, title = title, plot = False, \
                        to_check_ipw_mean1 = to_check_ipw_mean1_cutoffs, to_check_ipw_mean2 = to_check_ipw_mean2_cutoffs,\
                               reward_header = outcome, \
                               condition_header = condition_header, center = float(center.replace("pt","."))) #Note title not used here per say



#                    _, _, \
#                    _, _, \
#                     _ = hist_and_cutoffs(to_check = to_check, to_check_unif = to_check_unif, to_check_ipw = to_check_ipw, title_save = title_save,\
#    								       n = n, num_sims = num_sims, delta_nonzero = True, title = title, plot = True, \
#                        to_check_ipw_mean1 = to_check_ipw_mean1, to_check_ipw_mean2 = to_check_ipw_mean2,\
#                               reward_header = outcome, \
#                               condition_header = condition_header, center = float(center.replace("pt","."))) #Just for plotting, delta nonzero since passing in PCRS sims, also note not use to_check_ipw, it is wrong in this case, as not have delta, thus use means directly for wald stat for ipw 


                    print(cell_str)
                    print("percentile_dict_right")
                    print(percentile_dict_right)
    
                    title = "Type One Error Rates \n  n = {} and {} sims \n Initial Batch Size {} and Batch Size {}".format(n, num_sims, bs, bs)

                 #   next_col = stacked_bar_plot_with_cutoff(to_check = to_check,to_check_unif = to_check_unif,\
    		#			      n = n, num_sims = num_sims, \
    		#			       title = title,\
    		#			       percentile_dict_left = percentile_dict_left,\
    		#			       percentile_dict_right = percentile_dict_right,\
    		#			       ax = ax[i], bs_prop = bs_prop, ax_idx = i, outcomes_list = outcomes_list, reward_header = outcome, condition_header = condition_header)


                    next_col = stacked_bar_plot_with_cutoff(to_check = to_check,to_check_unif = to_check_unif, to_check_ipw = ipw_t1,\
					      n = n, num_sims = num_sims, \
					       title = title,\
					       percentile_dict_left = percentile_dict_left,\
					       percentile_dict_right = percentile_dict_right,\
                           percentile_dict_left_ipw =percentile_dict_left_ipw, \
                           percentile_dict_right_ipw = percentile_dict_right_ipw, \
					       ax = ax, bs_prop = bs_prop, ax_idx = i, \
                            outcomes_list = outcomes_list, reward_header = outcome, \
                             condition_header = condition_header, to_check_ipw_mean1 = to_check_ipw_mean1, to_check_ipw_mean2 = to_check_ipw_mean2)
		

                    next_table_dict["Batch size: " + str(bs_prop)] = next_col  
                    col.append(cell_str)

                df = pd.DataFrame(next_table_dict)
                df.index = ["Uniform","TS", "Empirical Cutoffs TS", "IPW", "Empirical Cutoffs IPW"]
                df.to_csv("Tables/Type1Error_For_{}_Experiment_With_{}_Outcome_p0_{}_p1_{}_n_{}_Center_{}.csv".format(experiment_title, outcome, str(p0).replace(".","pt"), str(p1).replace(".","pt"), n, center)) 

#                ax.set_title("Cutoffs From $p_1 = p_2 = {}$".format(float(center.replace("pt","."))))
                ax.set_ylabel("Power")
                #df["Cutoffs From p0 = p1 = {}".format(float(center.replace("pt",".")))] = col
               
 #df["Cutoffs From $p0 = p2 = {}$ \n PCRS Data with $p_0 = {}, p_1 = {}, |p_0 - p_1| = {}$".format(float(center.replace("pt",".")), p0, p1, delta)] = col
                i += 1
            


           # df.index = bs_prop_list
           # df = df.rename_axis("Arm Centers", axis="columns")
           # df = df.rename_axis("Batch Size", axis="rows")

          #  df.to_csv("Cutoffs/Cutoffs For_{}_Experiment_With_{}_Outcome_p0_{}_p1_{}_n_{}.csv".format(experiment_title, outcomes_title_dict[outcome], str(p0).replace(".","pt"), str(p1).replace(".","pt"), n))
            

          #  title_save = "Type One Error Rates Across {} Simulations For {} Experiment With {} Outcome p_0 = {}, p_1 = {}, |p_0 - p_1| = {} Number of Participants = {}".format(num_sims, experiment_title, outcomes_title_dict[outcome], p0, p1, delta, n)

            title_save = "Power Across {} Simulations For {} Experiment Number of Participants = {}".format(num_sims, experiment_title, n)


            title_save = title_save.replace(".","pt")
            title = "Power Across {} Simulations \n For {} Experiment \n \n Number of Participants = {}".format(num_sims, experiment_title, n)

            #ax[i].set_title(title, fontsize = 55)
            #i +=1
            #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
#            fig.suptitle(title)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    	 #   handles, labels = ax[i-1].get_legend_handles_labels()
    	  #  fig.legend(handles, labels, loc='upper right', prop={'size': 50})
        #fig.tight_layout()
            #handles, labels = ax[i-1].get_legend_handles_labels()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right') #prop={'size': 50})
#            fig.suptitle(title, fontsize = 23)
            fig.tight_layout(rect=[0, 0.03, 1, 0.82])
            fig.subplots_adjust(hspace = 0.5)
        #if not os.path.isdir("plots_multiarm_bs"):
         #   os.mkdir("plots_multiarm_bs")
            print("saving to ", "Singlecenter_plots/{}.png".format(title_save))
            fig.savefig("Singlecenter_plots/{}.png".format(title_save), bbox_inches = 'tight')
            plt.show()
            plt.clf()
            plt.close()


#REMEMBER TO CHANGE "/" to ":"
num_sims=5000

#Motivational Message ------
root = '../../Outfile/ParamsBased/numSims{}motivate_final/'
root_cutoffs = '../../Outfile/ParamsBased/numSims{}ForCutoffsCenters/'

sample_size_dict = {"Y1":348,"Y2":658,"Y3":478}
outcomes_title_dict = {"Y1": "Student success on the 2nd attempt at problem P1 for week 10", \
"Y2": "Student success on the 1st attempt at problem P2 for week 10", "Y3":"Student success on 1st attempt at problem P1 for week 12"}
experiment_title = "Motivational Message"
condition_header = "motivate_final"
#parse_dir(root, root_cutoffs, num_sims, outcomes_title_dict, sample_size_dict, experiment_title, condition_header)
#num_sims=5000
#Extra Problems-----------
root = '../../Outfile/ParamsBased/EffectnumSims{}problem_print_outputRedoV2/'
#root = '../../EffectnumSims5000problem_print_outputRedoV2/"

sample_size_dict = {"Y1":348,"Y2":658,"Y3":478, "Y4":657}
outcomes_title_dict = {"Y1": "Student success on the 2nd attempt at problem P1 for week 10", \
"Y2": "Student success on the 1st attempt at problem P2 for week 10", \
"Y3":"Student success on 1st attempt at problem P1 for week 12",\
"Y4":"Whether student succeeded in a maximum of 2 attempts at problem P3 on week 10"}



experiment_title = "Extra Problem"
condition_header = "problem_print_output"
parse_dir(root, root_cutoffs, num_sims, outcomes_title_dict, sample_size_dict, experiment_title, condition_header)



