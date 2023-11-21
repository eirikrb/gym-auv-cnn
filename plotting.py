import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from itertools import cycle


def plot_multiple_stats(df_list, label_list, var_list, var_labels_list, xaxis='episodes', window_size=100, n_timesteps = 1000000):
    plt.figure(figsize=(10, 6))
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
            smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std() 
            smoothed_var = gaussian_filter1d(df[var], sigma=150)
            #smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std()
            color = next(colors)['color'] # Get the next (first (hehe)) color from the cycle
            if xaxis == 'timesteps': 
                timesteps = np.arange(len(df[var])) * n_timesteps / len(df[var])
                plt.plot(timesteps, smoothed_var, color=color, label = label)
                #plt.plot(timesteps, df[var].to_numpy(), alpha=0.2, color=color)
                plt.fill_between(timesteps, (smoothed_var-smoothed_std).to_numpy(), (smoothed_var+smoothed_std).to_numpy(), alpha=0.2, color=color) #, label='Smoothed Std Dev')
                plt.xlabel('Timesteps',  fontsize=14)
                plt.legend(loc='best', fontsize=14)

            elif xaxis == 'episodes':
                plt.plot(smoothed_var,label = label, color=color)
                #plt.plot(df[var], alpha=0.2, color=color)
                plt.fill_between(df.index, smoothed_var-smoothed_std, smoothed_var+smoothed_std, alpha=0.2, color=color) #, label='Smoothed Std Dev')
                plt.xlabel('Episodes',  fontsize=14)
                plt.legend(loc='best', fontsize=14)

            plt.ylabel('Average Episode ' + ylabel_var.capitalize(), fontsize=14)
        
        plt.savefig(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/plots/{var}.pdf', bbox_inches='tight')
        plt.clf()
        plt.cla()

if __name__ == '__main__':
    # Add the dataframes to the list with the associated model configuration names
    #filenames = ['shallow_locked_stats', 'shallow_locked_beta_0.5_stats', 'shallow_locked_beta_1.5_stats', 'shallow_locked_beta_3.0_stats']#,'shallow_unlocked_stats', 'deep_locked_stats', 'deep_unlocked_stats']
    filenames = ['baseline_stats', 'shallow_locked_stats','shallow_unlocked_stats', 'deep_locked_stats', 'deep_unlocked_stats']
    df_list = [pd.read_csv(f'/home/eirikrb/Desktop/gym-auv-cnn/training_reports/data/{f}.csv') for f in filenames]
    #label_list = ['Shallow locked beta 1', 'Shallow locked beta 0.5', 'Shallow locked beta 1.5', 'Shallow locked beta 3.0']#, 'Shallow unlocked', 'Deep locked', 'Deep unlocked']
    label_list = ['Baseline', 'Shallow locked', 'Shallow unlocked', 'Deep locked', 'Deep unlocked']
    # Variables to plot
    var_list = ['rewards', 'progresses', 'cross_track_errors', 'timesteps', 'durations', 'collisions', 'goals_reached']
    var_labels_list = ['Reward', 'Progress', 'Cross track error', 'Timesteps', 'Duration', 'Collisions', 'Goals reached']
    
    window_size = 150 # set to same as sigma in gaussian_filter1d
    n_timesteps = 1000000 # should be same as in run.py

    # Note: When plotting multiple models on top of each other "episodes" makes them more comparable as every entry in the dataframe is an episode
    plot_multiple_stats(df_list=df_list,
                        label_list=label_list, 
                        var_list=var_list,
                        var_labels_list=var_labels_list,
                        window_size=window_size, 
                        n_timesteps=n_timesteps, 
                        xaxis='episodes')
    
