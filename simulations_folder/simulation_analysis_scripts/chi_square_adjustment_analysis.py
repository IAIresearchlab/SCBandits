import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
import os
from time import sleep
import subprocess

def get_all_file_paths(root_dir):
    to_return = []
    current_level_dfs = glob(f"{root_dir}/*Df.csv")
    if len(current_level_dfs) > 0:
        to_return += [df_path for df_path in current_level_dfs]
    else:
        for subdir in os.listdir(root_dir):
            full_dir = f"{root_dir}/{subdir}" 
            if os.path.isdir(full_dir):
                to_return += get_all_file_paths(full_dir)
    return to_return

if __name__ == "__main__":

    os.chdir("../simulation_scripts")
    os.system("./RunEffectSizeSimsSameArmsTSPPD.sh")
    os.chdir("../simulation_analysis_scripts")

    # wait 10 minutes for all the simulations to finish
    sleep(600)

    save_dir = 'TSPPDNoEffectResampleFast'

    # get chi square cutoffs for each combination of n and c
    num_sims = 500
    arm_prob = 0.5
    means = {}
    cutoffs = {}
    for n in (32, 88, 197, 785):
        for c in (0.025, 0.1, 0.2, 0.3, 0.05):
            try:
                file_path = glob(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*.csv")[0]
            except IndexError:
                print(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*.csv", "not found")
                continue
            df_sims = pd.read_csv(file_path)[:num_sims]
            plt.hist(df_sims['stat'])
            plt.title(f"Chi-Square Statistic: n={n}, c={c}")
            plt.xlabel("Chi-Square Statistic")
            plt.ylabel("# Sims")
            plt.savefig(f'../simulation_analysis_saves/chi_square_cutoff/chi_square_histogram_{n}.png')
            plt.show()
            cutoff = df_sims['stat'].sort_values().reset_index()['stat'][int(0.95 * num_sims)]
            print(f"cutoff: {cutoff}")
            cutoffs[f'n={n}_c={c}'] = cutoff
            print(f"chi square mean: {df_sims['stat'].mean()}")
            means[f'n={n}_c={c}'] = df_sims['stat'].mean()

    # delete the old simulations
    os.system(f"rm -rf {save_dir}")

    # re-run the simulations both with a difference between arms and no difference
    test_scripts = ["RunEffectSizeSimsSameArmsTSPPD.sh", "RunEffectSizeSimsTSPPD.sh"]
    os.chdir("../simulation_scripts")
    for test_script in test_scripts:
        os.system(f"./{test_script}")
    os.chdir("../simulation_analysis_scripts")

    # wait 10 minutes for the new simulations to finish
    sleep(600)

    # compute false positive rate
    save_dir = '../simulation_saves/TSPPDNoEffectResampleFast'
    num_sims = 500
    arm_prob = 0.5
    df_fp = pd.DataFrame()
    for n in (32, 88, 197, 785):
        for c in (0.025, 0.1, 0.2, 0.3, 0.05):
            try:
                file_path = glob(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*.csv")[0]
            except IndexError:
                print(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*.csv", "not found")
                continue
            df_sims = pd.read_csv(file_path)[:num_sims]
            if f'n={n}_c={c}' not in cutoffs:
                continue
            cutoff = cutoffs[f'n={n}_c={c}']
            df_positives = df_sims[df_sims['stat'] > cutoff]
            percent_positive = len(df_positives)/num_sims
            print(f"# above chi-square_cutoff: {len(df_positives)}")
            print(f"% of sims positive: {len(df_positives)/num_sims}")
            df_fp = df_fp.append({'effect_size': 0, 'n': n, 'c': c, 'percent_positive': percent_positive}, ignore_index=True)

    # compute true positive rate
    save_dir = '../simulation_saves/TSPPDIsEffectResampleFast'
    num_sims = 500
    arm_prob = 0.5
    df_power = pd.DataFrame()
    for es in (0.1, 0.2, 0.3, 0.5):
        for c in (0.025, 0.1, 0.2, 0.3, 0.05):
            try:
                file_path = glob(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/es={es}c={c}/*.csv")[0]
            except IndexError:
                print(save_dir + f"/num_sims={num_sims}armProb={arm_prob}/N={n}c={c}/*.csv", "not found")
                continue
            df_sims = pd.read_csv(file_path)
            for n in df_sims['num_steps'].drop_duplicates():
                if f'n={n}_c={c}' not in cutoffs:
                    continue
                cutoff = cutoffs[f'n={n}_c={c}']
                df_sims_n = df_sims[df_sims['num_steps'] == n][:num_sims]
                df_positives = df_sims_n[df_sims_n['stat'] > cutoff]
                percent_positive = len(df_positives)/num_sims
                print(f"# above chi-square_cutoff: {len(df_positives)}")
                print(f"% of sims positive: {len(df_positives)/num_sims}")
                df_power = df_power.append({'effect_size': es, 'n': n, 'c': c, 'percent_positive': percent_positive}, ignore_index=True)

    df_all = pd.concat([df_fp, df_power])
    df_all.to_csv('chi_square_adjustment_results.csv')