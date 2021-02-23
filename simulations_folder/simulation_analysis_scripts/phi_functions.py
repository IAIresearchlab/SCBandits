import ipdb as pb
import pandas as pd
import numpy as np
CUML_KEY = "prop_exploring_ppd_cuml"  
SNAPSHOT_KKEY = "exploring_ppd_at_this_n"

def plot_phi(df, num_sims, n, c, ax, es = 0):
    """
    get prop in cond 1 for when exploring
    """
#    pb.set_trace()
    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    
    extra_steps = [3,4,5]
    phi_list = [] #per step sizes
    phi_std_list = [] #per step sizes
   
    for num_steps in step_sizes:#loop over step sizes, make a hist for each
        print("num_steps", num_steps)
        
#        df[df["num_steps"] == 16]["exploring_ppd_at_this_n"].mean() 
        df_for_num_steps = df[df['num_steps'] == num_steps].dropna()

        phi = df_for_num_steps[SNAPSHOT_KKEY].mean() 
        phi_list.append(phi)

#    pb.set_trace()
    phi_list = np.array(phi_list)
    se = np.sqrt(phi_list*(1-phi_list))/np.sqrt(num_sims)
   # h = se * stats.t.ppf((1 + 0.95) / 2., num_sims-1)
   # h = stats.t.ppf(1-0.025, num_sims)*np.sqrt(phi_list*(1-phi_list)/num_sims) #95 CI for Proportion
    #print(phi_list, num_sims)
    ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
#  else:
  #      ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
    ind = np.arange(len(step_sizes))
    ind = step_sizes
    ax.set_xticks(ind)
    print(step_sizes)
    ax.set_xticklabels(step_sizes)
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0.0, 1.0)
    if n == 32:
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left')
        ax.legend(handles[::-1], labels[::-1])
    #leg2 = ax.legend(loc = 2)
    
