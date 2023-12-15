import pandas as pd
import scipy.stats as stats
import numpy as np

def read_csv_and_calculate_stats(file_path):
    '''Reads a csv file from file_path and calculates mean and 95% CI for each column. Result is only printed.'''
    df = pd.read_csv(file_path)
    print(f"File: {file_path}\n")
    # Iterate over each column to calculate mean and 95% CI
    for column in df.columns:
        if column == 'goals_reached': # Skip this column as all values are similar it throws an error for the ci
            continue
        data = df[column]
        mean = data.mean()
        ci_lower, ci_upper = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        print(f"{column}: Mean = {mean:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print('\n')


if __name__ == '__main__':
    
    # Run tests

    file_path_baseline = 'teststats_baseline.csv'  
    read_csv_and_calculate_stats(file_path_baseline)

    file_path_shallow = 'teststats_shallow.csv'
    read_csv_and_calculate_stats(file_path_shallow)

    file_path_deep = 'teststats_deep.csv'
    read_csv_and_calculate_stats(file_path_deep)