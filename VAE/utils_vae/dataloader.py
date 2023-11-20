from math import ceil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

N_SENSORS = 180

class LiDARDataset(torch.utils.data.Dataset):
  """Prepare the lidar observation dataset"""

  def __init__(self, 
               X:np.ndarray,  
               x_mean:np.float, x_std:np.float,
               prev_steps:int=None,
               standarize:bool=False):
    
    if not torch.is_tensor(X):
        if standarize:
            X = (X - x_mean)/x_std

    if prev_steps:
        X = prev_timesteps_feature_enginering(X, prev_steps)
    
    self.X = torch.Tensor(X[:,None]) # add channel dimension of size 1

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i]

def concat_csvs(csvs:tuple, output_path:str):
   """Stacks n csvs from list csvs containing paths into one csv at output_path"""
   arrs = tuple([np.loadtxt(csv) for csv in csvs])
   arr = np.concatenate(arrs, axis=0)
   np.savetxt(output_path, arr, delimiter=" ")


def apply_random_rotations(X:np.ndarray, n_rotations:int=10):
    """Apply n_rotations random rotations in the range [-180,180] deg to X"""
    X_rotated = X.copy()
    for i in range(n_rotations):
        roll = np.random.randint(-180, 180)
        roll_int = int(roll/2)
        # Get random sample from X, rotate it and add it to X_rotated
        index = np.random.randint(0, len(X))
        X_samples = X[index:index+2,:]
        X_rotated = np.concatenate((X_rotated, np.roll(X_samples, roll, axis=1)), axis=0)
    return X_rotated

def load_LiDARDataset(path_x:str, 
                      batch_size:int, 
                      mode:str=None, 
                      prev_steps:bool=None,
                      train_test_split:float=0.7, 
                      train_val_split:float=0.2, 
                      shuffle:bool=True,
                      extend_dataset_roll:bool=False,
                      roll_degrees:list=[-90,90],
                      num_rotations:int=1000,
                      add_noise_to_train:bool=False) -> tuple:
    """
    Load data, split into train, validation and test sets, and create dataloaders.
    - Possible to extend dataset by rolling (rotating) the data
    - Possible to add noise to the training set.
    """

    X = np.loadtxt(path_x)
    X = 1 - X/150

    if extend_dataset_roll:
        #X = apply_random_rotations(X, n_rotations=num_rotations)
        #'''
        for roll_deg in roll_degrees:
            roll_int = int(roll_deg/2)
            X = extend_dataset_rolling(X=X, roll=roll_int)
        #'''
    data_size  = len(X)
    train_size = int(train_test_split * data_size)
    val_size   = int(train_val_split * train_size)
    test_size  = data_size - train_size
    train_size = train_size - val_size
    
    # Training set
    X_train = X[:train_size,:]

    if add_noise_to_train:
        X_train = X_train + np.random.normal(0, 0.007, size=X_train.shape) # Suitable noise variance found by plotting
    
    x_mean, x_std = X_train.mean(), X_train.std() # Mean and std only from training set
    
    data_train = LiDARDataset(X_train, x_mean, x_std, prev_steps=prev_steps)
    dataloader_train = torch.utils.data.DataLoader(data_train, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle, 
                                                num_workers=1,
                                                drop_last=True)

    # Validation set
    X_val = X[train_size:train_size+val_size,:]
    data_val = LiDARDataset(X_val, x_mean, x_std, prev_steps=prev_steps)
    dataloader_val = torch.utils.data.DataLoader(data_val, 
                                                 batch_size=batch_size, 
                                                 shuffle=shuffle, 
                                                 num_workers=1,
                                                 drop_last=True)
    # Test set
    X_test = X[-test_size:,:]
    data_test = LiDARDataset(X_test, x_mean, x_std, prev_steps=prev_steps)  
    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=1, 
                                                  shuffle=shuffle, 
                                                  num_workers=1,
                                                  drop_last=True)
    
    return data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test


def prev_timesteps_feature_enginering(X:np.ndarray, time_steps:int):
    X_concat = X[:,:,None].copy()
    X_prev   = X.copy()
    x_empty  = np.array([150]*N_SENSORS).reshape((1, N_SENSORS))
    for i in range(time_steps):
        X_prev = X_prev[:-1,:].copy()
        X_prev = np.concatenate((x_empty, X_prev), axis=0)
        X_concat = np.concatenate((X_concat, X_prev[:,:,None]), axis=2)
    return X_concat
    

def extend_dataset_rolling(X:np.ndarray, roll:int=90) -> np.ndarray:
    """Extend the dataset by rolling the data by a "roll" number of steps (two degrees per one step)."""
    X_extended = X.copy()
    X_extended = np.concatenate((X_extended, np.roll(X, roll, axis=1)), axis=0)
    return X_extended
