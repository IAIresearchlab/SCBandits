'''
Created on Jul 31, 2017

@author: rafferty
'''
import matplotlib
matplotlib.use('PDF')
import numpy as np
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
import run_effect_size_simulations
rcParams.update({'figure.autolayout': True})
import pandas as pd
import sys

TRIAL_START_NUM = 10

def make_overall_stats_df(df, arm_stats = None, assume_t = False, effect_size=None):
    step_sizes = df['num_steps'].unique()
    rows = []
    for num_steps in step_sizes:
        cur_row = {}
        df_for_num_steps = df[df['num_steps'] == num_steps]
        cur_row['num_steps'] = num_steps
        num_replications = len(df_for_num_steps)
        cur_row['num_reps'] = num_replications
        num_rejected = np.sum(df_for_num_steps['pvalue'] < .05)
        cur_row['proportion_null_rejected'] = num_rejected / num_replications
        cur_row['avg_ratio'] = np.mean(df_for_num_steps['ratio'])
        cur_row['mean_1'] = np.mean(df_for_num_steps['mean_1'])
        cur_row['mean_2'] = np.mean(df_for_num_steps['mean_2'])
        cur_row['min_cond_1'] = df_for_num_steps.min()['mean_1']
        cur_row['min_cond_2'] = df_for_num_steps.min()['mean_2']
        cur_row['max_cond_1'] = df_for_num_steps.max()['mean_1']
        cur_row['max_cond_2'] = df_for_num_steps.max()['mean_2']
        if assume_t and effect_size != None:
            expected_power = smp.tt_ind_solve_power(effect_size, num_steps / 2, 0.05, None, 1)
            cur_row['expected_power'] = expected_power
        avg_es = np.mean(df_for_num_steps['actual_es'])
        cur_row['avg_es'] = avg_es
        count_significant_cases = np.count_nonzero(df_for_num_steps['pvalue'] < .05)
        count_sign_errors_given_significant = np.count_nonzero((df_for_num_steps['pvalue'] < .05) & (df_for_num_steps['mean_1'] < df_for_num_steps['mean_2']))
        count_sign_errors_overall = np.count_nonzero(df_for_num_steps['mean_1'] < df_for_num_steps['mean_2'])
        if count_significant_cases != 0:
            cur_row['type_s_errors'] = count_sign_errors_given_significant / count_significant_cases
        else:
            cur_row['type_s_errors'] = 0
        cur_row['overall_prop_sign_reversed'] = count_sign_errors_overall / num_replications
        rows.append(cur_row)
        
    # Make dataframe
    df = pd.DataFrame(rows)
    
    return df

def print_output_stats(df, arm_stats = None, assume_t = False, effect_size=None, prior_params = None, reordering_info = None):
    '''Assumes a pandas data frame as calculated by calculate_statistics_from_sims in a module prefixed
    with run_effect_size_simulations. Prints out various stats related to the simulations. If assume_t = True,
    then expected_power is calculated based on a ttest. Otherwise, expected power is calculated based on
    a 2x2 chi square contingency test. 
    '''
    step_sizes = df['num_steps'].unique()
    output_string = '';
    for num_steps in step_sizes:
        df_for_num_steps = df[df['num_steps'] == num_steps]
        wald_pval_for_num_steps = df_for_num_steps['wald_pval'].mean()
        wald_stat_for_num_steps = df_for_num_steps['wald_type_stat'].mean()
        pval_for_num_steps = df_for_num_steps['pvalue'].mean()

        num_replications = len(df_for_num_steps)
        output_string += '\n'
        output_string += 'Total number of samples:' + str(num_steps) + '\n'
        if arm_stats != None:
            output_string += 'Arm stats: ' + str(arm_stats) + '\n'
        if prior_params != None:
            output_string += 'Prior params: ' + str(prior_params) + '\n'
        if reordering_info != None:
            output_string += 'Reordering info: ' + str(reordering_info) + '\n'
        num_rejected = np.sum(df_for_num_steps['pvalue'] < .05)
        num_rejected_wald = np.sum(df_for_num_steps['wald_pval'] < .05)

        output_string += 'Proportion of samples where null hyp was rejected:' + str(num_rejected / num_replications) + '\n'
        output_string += 'Proportion of samples where null hyp was rejected based on Wald-type statistic: ' + str(num_rejected_wald / num_replications) + '\n'
        output_string += 'P value from Wald-type statistic: ' + str(wald_pval_for_num_steps) + '\n'
        output_string += 'Wald-type statistic: ' + str(wald_stat_for_num_steps) + '\n'
        output_string += 'P value: ' + str(pval_for_num_steps) + '\n'
        
        output_string += 'Average ratio of condition1:condition2: ' + str(np.mean(df_for_num_steps['ratio'])) + '\n'
        output_string += 'Mean condition 1: ' + str(np.mean(df_for_num_steps['mean_1'])) + '\n'
        output_string += 'Mean condition 2: ' + str(np.mean(df_for_num_steps['mean_2'])) + '\n'
        range_condition1 = [df_for_num_steps.min()['mean_1'],df_for_num_steps.max()['mean_1']]
        output_string += 'Range condition 1: ' + str(range_condition1 ) + '\n'
        range_condition2 = [df_for_num_steps.min()['mean_2'],df_for_num_steps.max()['mean_2']]
        output_string += 'Range condition 2: ' + str(range_condition2) + '\n'
        if assume_t and effect_size != None:
            expected_power = smp.tt_ind_solve_power(effect_size, num_steps / 2, 0.05, None, 1)
            output_string += 'Expected power for sample size (equal ratio):' + str(expected_power) + '\n'
        avg_es = np.mean(df_for_num_steps['actual_es'])
        output_string += 'Average actual effect size: ' + str(avg_es) + '\n'
        output_string += 'Average power based on ratio:' + str(np.mean(df_for_num_steps['power'])) + '\n'
        count_significant_cases = np.count_nonzero(df_for_num_steps['pvalue'] < .05)
        output_string += "Number of significant cases: " + str(count_significant_cases) + "\n"
        count_sign_errors_given_significant = np.count_nonzero((df_for_num_steps['pvalue'] < .05) & (df_for_num_steps['mean_1'] < df_for_num_steps['mean_2']))
        count_sign_errors_overall = np.count_nonzero(df_for_num_steps['mean_1'] < df_for_num_steps['mean_2'])
        if count_significant_cases != 0:
            output_string += 'Proportion of significant cases where sign is reversed:' + str(count_sign_errors_given_significant / count_significant_cases) + '\n'
        else:
            output_string += 'Proportion of significant cases where sign is reversed:' + str(0) + '\n'

        output_string += 'Proportion of all cases where sign is reversed:' + str(count_sign_errors_overall / num_replications) + '\n'

    return output_string

def plot_power_by_steps(df_by_trial, alpha = 0.05, target_power = 0.8):
    '''
    df_by_trial is a data frame with information about each run, as calculated by calculate_by_trial_statistics_from_sims.
    This pops up a line graph of how the power changes with number of trials. Power is calculated as what proportion of the p-values
    were below alpha at that point.
    '''
    unique_sample_sizes = df_by_trial.num_steps.unique()
    figure = plt.figure(figsize = (8, 2))
#     matplotlib.rcParams.update(params)
    for i in range(len(unique_sample_sizes)):
        cur_n = unique_sample_sizes[i]
        cur_df = df_by_trial[df_by_trial['num_steps'] == cur_n]
        statistic_list = []
        for trial in range(TRIAL_START_NUM, cur_n):
            avg_stat = np.sum(cur_df[cur_df['trial'] == trial]['pvalue'] < .05) / len(cur_df[cur_df['trial'] == trial])
            statistic_list.append(avg_stat)
        
        # Now add the figure and plot
        x = range(TRIAL_START_NUM, cur_n)
        ax = figure.add_subplot(1,len(unique_sample_sizes), i + 1)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        plt.plot(x, statistic_list, lw=2)
        plt.xlabel('Trial Num', fontsize = 8)
        if i == 0:
            plt.ylabel('Power')
        
        plt.plot(x, target_power*np.ones(len(x)), '--')
    return figure

def make_hist_of_trials(df):
    '''
    df is a data frame with information about each run, as calculated by calculate_statistics_from_sims.
    This pops up a histogram where bins are based on the proportion of trials that were arm 1.
    '''
    unique_sample_sizes = df.num_steps.unique()
    figure = plt.figure()
    for i in range(len(unique_sample_sizes)):
        cur_n = unique_sample_sizes[i]
        proportions = df[df['num_steps'] == cur_n]['sample_size_1'] / cur_n
        
        new_plot = figure.add_subplot(1,len(unique_sample_sizes), i + 1)
        bins = np.arange(0, 1.01, .025)
        n, bins, patches = new_plot.hist(proportions, bins)
        for item in patches:
            item.set_height(item.get_height()/sum(n))
        if i == 0:
            plt.ylabel('Probability')
        plt.xlabel('Prop. of trials cond1', fontsize = 8)
        plt.title('num steps ' + str(cur_n))
        plt.axis([0, 1.05, 0, 1])

    return figure

def make_by_trial_arm_statistics(arm_df_by_trial, num_arms):
    '''
    arm_df_by_trial is a data frame output by create_arm_stats_by_step that has the means and variances of 
    each arm at every step. This function creates a figure with two plots: one showing how the mean estimate changes
    over time (mean of the means, with 1 standard error) and one showing how the variance estimate changes over
    time (mean of the variances, with 1 standard error).
    '''
    sample_size = arm_df_by_trial.num_steps.unique()[0]
    figure = plt.figure(figsize = (8, 2))
    mean_stats = np.zeros((sample_size, num_arms))
    var_stats = np.zeros((sample_size, num_arms))
    mean_se_stats = np.zeros((sample_size, num_arms))
    var_se_stats = np.zeros((sample_size, num_arms))
    for trial in range(TRIAL_START_NUM, sample_size):
        mean_stats[trial,] = [np.mean(arm_df_by_trial[arm_df_by_trial['trial'] == trial]['mean_arm_' + str(i)]) for i in range(1, num_arms+1)]
        var_stats[trial,] = [np.mean(arm_df_by_trial[arm_df_by_trial['trial'] == trial]['var_arm_' + str(i)]) for i in range(1, num_arms+1)]
        mean_se_stats[trial,] = [stats.sem(arm_df_by_trial[arm_df_by_trial['trial'] == trial]['mean_arm_' + str(i)]) for i in range(1, num_arms+1)]
        var_se_stats[trial,] = [stats.sem(arm_df_by_trial[arm_df_by_trial['trial'] == trial]['var_arm_' + str(i)]) for i in range(1, num_arms+1)]
    # Now add the figure and plot
    x = range(TRIAL_START_NUM, sample_size)
    ax = figure.add_subplot(1,2, 1)
    plot_val_and_errors(ax, x, mean_stats, mean_se_stats, 'Arm Mean')
    ax = figure.add_subplot(1,2, 2)
    plot_val_and_errors(ax, x, var_stats, var_se_stats, 'Arm Variance')
    return figure
        
def plot_val_and_errors(ax, xs, value, ses, ylabel):
    legend_labels = []
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    for arm_index in range(np.size(value,1)):
        legend_labels.append('arm ' + str(arm_index + 1))
        error_above = value[:,arm_index] + ses[:,arm_index]
        error_below = value[:,arm_index] - ses[:,arm_index]
        plt.fill_between(xs, error_below[TRIAL_START_NUM:], error_above[TRIAL_START_NUM:], interpolate=True, alpha = 0.5)
        plt.plot(xs, value[TRIAL_START_NUM:, arm_index], lw=2)
    plt.xlabel('Trial Num', fontsize = 8)
    plt.ylabel(ylabel)

    

def make_by_trial_graph_of_column(df_by_trial, column_name):
    '''
    df_by_trial is a data frame with information about each run, as calculated by calculate_by_trial_statistics_from_sims.
    This pops up a line graph of how the value of the column labeled column_name changes across trials. The mean is plotted,
    with error bars showing 1 standard error above and below. 
    '''
    unique_sample_sizes = df_by_trial.num_steps.unique()
    figure = plt.figure(figsize = (8, 2))
#     matplotlib.rcParams.update(params)
    for i in range(len(unique_sample_sizes)):
        cur_n = unique_sample_sizes[i]
        cur_df = df_by_trial[df_by_trial['num_steps'] == cur_n]
        statistic_list = []
        errors = []
        for trial in range(TRIAL_START_NUM, cur_n):
            avg_stat = np.mean(cur_df[cur_df['trial'] == trial][column_name])
            statistic_list.append(avg_stat)
            #std_error = stats.sem(cur_df[cur_df['trial'] == trial][column_name])
            std_error = np.std(cur_df[cur_df['trial'] == trial][column_name])

            errors.append(std_error)
        
        # Now add the figure and plot
        x = range(TRIAL_START_NUM, cur_n)
        ax = figure.add_subplot(1,len(unique_sample_sizes), i + 1)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        error_above = np.array(statistic_list) + np.array(errors)
        error_below = np.array(statistic_list) - np.array(errors)
        plt.fill_between(x, error_below, error_above, interpolate=True, alpha = 0.5)
        plt.plot(x, statistic_list, lw=2)
        plt.xlabel('Trial Num', fontsize = 8)
        if i == 0:
            plt.ylabel(column_name)
        if column_name == "pvalue":
            plt.plot(x, run_effect_size_simulations.DESIRED_ALPHA*np.ones(len(x)), '--')
    return figure

def add_expected_reward_to_figure(figure, true_arm_means, num_steps):
    '''
    Plots the rewards we receive on a trial by trial basis compared to the expected reward at that point
    if we were completely balancing the experiment.
    '''
    avg_reward = np.mean(true_arm_means)
    
    for i, ax in enumerate(figure.axes):
        x = np.arange(TRIAL_START_NUM, num_steps[i])
        ax.plot(x, avg_reward*x, '--')
    return figure

def make_type_s_plot(overall_stats_df_list, effect_sizes):
    figure = plt.figure(figsize = (8, 2))
    for i in range(len(effect_sizes)):
        cur_es = effect_sizes[i]
        cur_df = overall_stats_df_list[i]
        
        all_num_steps = cur_df.num_steps.unique()
        ax = figure.add_subplot(1,len(effect_sizes), i + 1)
        
        ind = np.arange(len(all_num_steps))  # the x locations for the groups
        
        
        width = 0.35       # the width of the bars
        
        rects1 = ax.bar(ind, cur_df['type_s_errors'], width, label='Type S proportion')
        rects2 = ax.bar(ind + width, cur_df['overall_prop_sign_reversed'], width, color='C2', label='Overall sign reversal proportion')
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Number of Steps')
        ax.set_title('Sign Reversals: Effect Size = ' + str(cur_es))
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(cur_df['num_steps'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return figure
        

def main():
    effect_sizes = [float(es) for es in sys.argv[1].split(',')]
    outfile_directory = sys.argv[2]
    bandit_type_prefix = sys.argv[3]
    
    overall_stats_dfs = [pd.read_pickle(outfile_directory + bandit_type_prefix + str(es) + 'OverallStatsDf.pkl') for es in effect_sizes]
    type_s_figure = make_type_s_plot(overall_stats_dfs, effect_sizes)
    type_s_figure.savefig(outfile_directory + bandit_type_prefix + 'TypeSFigure.pdf', bbox_inches='tight')
    
if __name__=="__main__":
    main()
    


        