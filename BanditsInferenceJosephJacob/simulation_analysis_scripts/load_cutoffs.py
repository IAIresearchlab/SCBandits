import glob
import numpy as np
import ipdb
import pickle
from rectify_vars_and_wald_functions import *
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
def load_cutoffs(n, num_steps, arm_prob):
    bs=1


    n_dir = n 
    if n == 657/4:
        n_dir = 657
    print("loading num_steps", num_steps, "n", n)
    num_sims_ts = 10000
    root_ts = "../simulation_saves/TSPPDNoEffectFast/num_sims={}armProb={}".format(num_sims_ts, arm_prob)
    root_ts_jeff = "../simulation_saves/TSPPDNoEffectFastJeffPrior/num_sims={}armProb={}".format(num_sims_ts, arm_prob)
    ts_dir = root_ts + "/N={}c=0.0/".format(n_dir)
    ts_jeff_dir = root_ts_jeff + "/N={}c=0.0/".format(n_dir)

    print(ts_dir)
    to_check_ts= glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
    assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

    to_check_ts_jeff = glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
    assert(len(glob.glob(ts_jeff_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

    with open(to_check_ts, 'rb') as t:
        df_ts = pickle.load(t)
      
    with open(to_check_ts_jeff, 'rb') as t:
        df_ts_jeff = pickle.load(t)
      
   # ipdb.set_trace()
    rect_key = "Drop NA"
    #rect_key = "TS"
    rectify_vars_noNa(df_ts, alg_key = rect_key)
    rectify_vars_noNa(df_ts_jeff, alg_key = rect_key)
    es = 0

    df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()

    df_for_num_steps_ts_cutoffs = df_for_num_steps_ts[5000:]
    if n_dir == 657:
        df_for_num_steps_ts_cutoffs = df_for_num_steps_ts
    df_for_num_steps_ts = df_for_num_steps_ts[0:5000]
    wald_type_stat_co = df_for_num_steps_ts_cutoffs['wald_type_stat']

    #ipdb.set_trace()
    ts_cutoffs_L = np.percentile(wald_type_stat_co, 2.5)
    ts_cutoffs_R = np.percentile(wald_type_stat_co, 97.5)
    ts_cutoffs = [ts_cutoffs_L, ts_cutoffs_R]

    df_for_num_steps_ts_jeff = df_ts_jeff[df_ts_jeff['num_steps'] == num_steps].dropna()
    df_for_num_steps_ts_jeff_cutoffs = df_for_num_steps_ts_jeff[5000:]

    if n_dir == 657:
        df_for_num_steps_ts_jeff_cutoffs = df_for_num_steps_ts_jeff

    df_for_num_steps_ts_jeff = df_for_num_steps_ts_jeff[0:5000]
    wald_type_stat_co_jeff = df_for_num_steps_ts_jeff_cutoffs['wald_type_stat']

    ts_cutoffs_jeff_L = np.percentile(wald_type_stat_co_jeff, 2.5)
    ts_cutoffs_jeff_R = np.percentile(wald_type_stat_co_jeff, 97.5)
    ts_cutoffs_jeff = [ts_cutoffs_jeff_L, ts_cutoffs_jeff_R]

    return ts_cutoffs, ts_cutoffs_jeff

