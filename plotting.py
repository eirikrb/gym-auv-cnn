import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from itertools import cycle


def plot_multiple_stats(df_list, label_list, var_list, var_labels_list, xaxis='episodes', window_size=100, n_timesteps = 1000000, path:str=None, legend_plot:str='rewards'):
    plt.figure(figsize=(10, 8))
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    for i, var in enumerate(var_list):
        ylabel_var = var_labels_list[i]
        # Ensuring same color-stuff:
        current_cycler = plt.rcParams['axes.prop_cycle'] # Retrieve the current color cycle
        colors = cycle(current_cycler)
        for j in range(len(df_list)):
            df = df_list[j]
            label = label_list[j] 
            # calculate the smoothed var and the std using a moving average
            #smoothed_var = df[var].rolling(window_size, center = True, min_periods=1).mean()
            #smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std() 
            smoothed_var = gaussian_filter1d(df[var], sigma=window_size)
            #smoothed_std = gaussian_filter1d(df[var].rolling(window_size, center = True, min_periods=1).std(),sigma=window_size/2) 
            smoothed_std = df[var].rolling(2*window_size, center = True, min_periods=1).std()
            #smoothed_var = df[var].ewm(span=window_size, ignore_na=True).mean()
            #smoothed_std = df[var].ewm(span=window_size).std()
            #smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std()
            color = next(colors)['color'] # Get the next (first (hehe)) color from the cycle
            linestyle = 'solid'
            if label == 'PPO + 1-layer CNN [unlocked] (Baseline)':
                color = 'black'
                linestyle = 'dashed'
            if xaxis == 'timesteps': 
                timesteps = np.arange(len(df[var])) * n_timesteps / len(df[var])
                plt.plot(timesteps, smoothed_var, color=color, label=label, linestyle=linestyle, linewidth=1)
                #plt.plot(timesteps, df[var].to_numpy(), alpha=0.2, color=color)
                plt.fill_between(timesteps, (smoothed_var-smoothed_std).to_numpy(), (smoothed_var+smoothed_std).to_numpy(), alpha=0.2, color=color) #, label='Smoothed Std Dev')
                plt.xlabel('Timesteps',  fontsize=14)
                plt.legend(loc='best', fontsize=14)

            elif xaxis == 'episodes':
                plt.plot(smoothed_var,label=label, color=color, linestyle=linestyle, linewidth=1.5)
                #plt.plot(df[var], alpha=0.2, color=color)
                #plt.fill_between(df.index, smoothed_var-smoothed_std, smoothed_var+smoothed_std, alpha=0.2, color=color) #, label='Smoothed Std Dev')
                plt.xlabel('Episodes',  fontsize=14)
                if var == legend_plot:
                    plt.legend(loc='best', fontsize=20)

            plt.ylabel('Avg. Episode ' + ylabel_var, fontsize=14)
        
        plt.savefig(f'{path}/{var}.pdf', bbox_inches='tight')
        plt.clf()
        plt.cla()

if __name__ == '__main__':  
    
    window_size = 150 
    n_timesteps = 1000000 # should be same as in run.py
    var_list = ['rewards', 'progresses', 'cross_track_errors', 'timesteps', 'durations', 'collisions', 'goals_reached']
    var_labels_list = ['Reward', 'Progress', 'CTE', 'Timesteps', 'Duration', 'Collisions', 'Goals reached']

    # PLOTTING 4 CONFIGURATION COMPARISON 1M timesteps
    # Note: When plotting multiple models on top of each other "episodes" makes them more comparable as every entry in the dataframe is an episode
    '''
    filenames = ['shallow_locked_stats','shallow_unlocked_stats', 'deep_locked_stats', 'deep_unlocked_stats', 'nosafety_stats']
    df_list = [pd.read_csv(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/data/basetest_1M/{f}.csv') for f in filenames]
    label_list = ['PPO + ShallowVAE [locked]', 'PPO + ShallowVAE [unlocked]', 'PPO + DeepVAE [locked]', 'PPO + DeepVAE [unlocked]', 'No safety filter']
    save_path = '/home/eirikrb/Desktop/gym-auv-cnn/training_reports/plots'#/4_config_comparison'
    plot_multiple_stats(df_list=df_list,
                        label_list=label_list, 
                        var_list=var_list,
                        var_labels_list=var_labels_list,
                        window_size=window_size, 
                        n_timesteps=n_timesteps, 
                        xaxis='episodes',
                        path=save_path,
                        legend_plot='cross_track_errors') # legend_plot denotes which plot to have the legend in (upper right in report)
    '''
    '''
    # PLOTTING 4 CONFIGURATION COMPARISON 3M timesteps
    filenames = ['shallow_locked_3M_stats','shallow_unlocked_3M_stats', 'deep_locked_3M_stats', 'deep_unlocked_3M_stats']#, 'baseline_3M_stats']
    df_list = [pd.read_csv(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/data/basetest_3M/{f}.csv') for f in filenames]
    label_list = ['PPO + ShallowVAE [locked]', 'PPO + ShallowVAE [unlocked]', 'PPO + DeepVAE [locked]', 'PPO + DeepVAE [unlocked]']#, 'Baseline']
    save_path = '/home/eirikrb/Desktop/gym-auv-cnn/training_reports/plots/4_config_comparison_3M'
    plot_multiple_stats(df_list=df_list,
                        label_list=label_list, 
                        var_list=var_list,
                        var_labels_list=var_labels_list,
                        window_size=window_size, 
                        n_timesteps=n_timesteps, 
                        xaxis='episodes',
                        path=save_path,
                        legend_plot='cross_track_errors') # legend_plot denotes which plot to have the legend in (upper right in report)
    '''
    '''
    # PLOTTING SHALLOW LOCKED VS BASELINE
    filenames = ['shallow_locked_3M_stats','baseline_3M_stats']
    df_list = [pd.read_csv(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/data/basetest_3M/{f}.csv') for f in filenames]
    label_list = ['PPO + ShallowVAE [locked] (Ours)', 'PPO + 1-layer CNN [unlocked] (Baseline)']
    save_path = '/home/eirikrb/Desktop/gym-auv-cnn/training_reports/plots/baselinetest_3M'
    plot_multiple_stats(df_list=df_list,
                        label_list=label_list, 
                        var_list=var_list,
                        var_labels_list=var_labels_list,
                        window_size=window_size, 
                        n_timesteps=n_timesteps, 
                        xaxis='episodes',
                        path=save_path,
                        legend_plot='cross_track_errors') # legend_plot denotes which plot to have the legend in (upper right in report)
    '''
    #'''
    # BETA COMPARISON
    label_list = ['β = 0', 'β = 0.1',  'β = 0.5', 'β = 1.0', 'β = 1.5', 'β = 3.0']
    filenames = ['shallow_locked_beta_0.0_stats', 'shallow_locked_beta_0.1_stats', 'shallow_locked_beta_0.5_stats', 'shallow_locked_stats',  'shallow_locked_beta_1.5_stats', 'shallow_locked_beta_3.0_stats']
    df_list = [pd.read_csv(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/data/betatest/{f}.csv') for f in filenames]
    save_path = '/home/eirikrb/Desktop/gym-auv-cnn/training_reports/plots/beta_test'
    plot_multiple_stats(df_list=df_list,
                        label_list=label_list, 
                        var_list=var_list,
                        var_labels_list=var_labels_list,
                        window_size=window_size, 
                        n_timesteps=n_timesteps, 
                        xaxis='episodes',
                        path=save_path,
                        legend_plot='cross_track_errors') # legend_plot denotes which plot to have the legend in (upper right in report)
    #'''
