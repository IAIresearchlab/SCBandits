import numpy as np
import scipy.stats
def rectify_vars_Na(df):
    '''
    pass in those which have NA wald
    '''
    assert (np.sum(df["sample_size_1"] == 0) + np.sum(df["sample_size_2"] == 0)) == np.sum(df["wald_type_stat"].isna())


    total_reward = df["total_reward"] #know total reward is all in the arm with samples

    df_empty_arm_1 = df[df["sample_size_1"] == 0] #sims which have empty arm 1
    df["mean_1"].loc[df["sample_size_1"] == 0] = 0.5
    df["mean_2"].loc[df["sample_size_1"] == 0] = (df_empty_arm_1["total_reward"] + 0.5)/(df_empty_arm_1["sample_size_2"] + 1) 

    df_empty_arm_2 = df[df["sample_size_2"] == 0] #sims which have empty arm 1
    df["mean_2"].loc[df["sample_size_2"] == 0] = 0.5
    df["mean_1"].loc[df["sample_size_2"] == 0] = (df_empty_arm_2["total_reward"] + 0.5)/(df_empty_arm_2["sample_size_1"] + 1) 
#    return df

def rectify_vars_noNa(df, alg_key = "TS"):
    """
    the formula should do fine, zeros out the nans in means, deal with wald nans next
    """
#    assert np.sum(df["wald_type_stat"].isna()) == 0
#    assert (np.sum(df["sample_size_1"] == 0) + np.sum(df["sample_size_2"] == 0)) == np.sum(df["wald_type_stat"].isna())
  
#    df.loc[df["sample_size_1"] == 0, "mean_1"] = -99 #should get zerod out in the followning
 #   df.loc[df["sample_size_2"] == 0, "mean_2"] = -99 #should get zerod out in the followning

    df["mean_1"].loc[df["sample_size_1"] == 0] = -99 #should get zerod out in the followning
    df["mean_2"].loc[df["sample_size_2"] == 0] = -99 #should get zerod out in the followning

    if alg_key == "EG":
        df["mean_1"] = (df["mean_1"]*df["sample_size_1"] + 0.5)/(df["sample_size_1"] + 1)
        df["mean_2"] = (df["mean_2"]*df["sample_size_2"] + 0.5)/(df["sample_size_2"] + 1)
        df["sample_size_1"] = df["sample_size_1"] + 2 #two missing sample per arm, 1 success 1 failure from prior
        df["sample_size_2"] = df["sample_size_2"] + 2
        compute_wald(df)
    if alg_key == "Drop NA":
        df.dropna(inplace = True)
    if alg_key == "TS":
        df["mean_1"] = (df["mean_1"]*df["sample_size_1"] + 1.0)/(df["sample_size_1"] + 2)
        df["mean_2"] = (df["mean_2"]*df["sample_size_2"] + 1.0)/(df["sample_size_2"] + 2)
    
        df["sample_size_1"] = df["sample_size_1"] + 2 #two missing sample per arm, 1 success 1 failure from prior
        df["sample_size_2"] = df["sample_size_2"] + 2

        compute_wald(df)
    assert np.sum(df["wald_type_stat"].isna()) == 0
    assert np.sum(df["wald_pval"].isna()) == 0

def compute_wald(df, delta = 0):
    mean_1, mean_2, sample_size_1, sample_size_2 = df["mean_1"], df["mean_2"], df["sample_size_1"], df["sample_size_2"]

    SE = np.sqrt(mean_1*(1 - mean_1)/sample_size_1 + mean_2*(1 - mean_2)/sample_size_2) #will be 0 if get all 1's for arm1, and 0 for arm2
    wald_type_stat = (mean_1 - mean_2)/SE #(P^hat_A - P^hat_b)/SE
    #print('wald_type_stat:', wald_type_stat)
    wald_pval = (1 - scipy.stats.norm.cdf(np.abs(wald_type_stat)))*2 #Two sided, symetric, so compare to 0.05
    df["wald_type_stat"] = wald_type_stat
    df["wald_pval"] = wald_pval
