import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mutual_info_score
import numpy as np
import os
import argparse
import json


# UNFINISHED MODULE AS PER DEC 2023

class Disentanglement_evaluator():
    """
    Class for measuring  the disentanglement of the latent space, given some model. (Both) metric (w/o name) (and mutual information gap (MIG)) are implemented
    See https://openreview.net/pdf?id=Sy2fzU9gl chapter 3 for more details on metric w/o name and 
    https://arxiv.org/pdf/1802.04942.pdf for more details on MIG
    """
    def __init__(self, 
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 dataloader_test:torch.utils.data.Dataloader,
                 model:nn.Module,
                 batch_size:int) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test
        self.model = model
        self.batch_size = batch_size

        self.latent_factor_indeces = list(range(6)) # fINN UT HVA DENNE SKAL VÆRE, essentially K in pseudocode(?)


    def create_training_sample(self):
        n_factors = len(self.latent_factor_indeces)
        y = np.random.randint(n_factors) # y ~ Unif[1...K]



    def create_training_batch(self, n_samples):
        points = None
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            labels[i]

    def _sample_observations_from_factors(factors, latent_factor_indices, factor_sizes, images):
        num_samples = factors.shape[0]
        num_factors = len(latent_factor_indices)

        all_factors = np.zeros(shape=(num_samples, num_factors), dtype=np.int64)
        all_factors[:, latent_factor_indices] = factors

        # Complete all the other factors
        observation_factor_indices = [
            i for i in range(num_factors) if i not in latent_factor_indices
        ]

        for i in observation_factor_indices:
            all_factors[:, i] = np.random.randint(factor_sizes[i], size=num_samples)

        factor_bases = np.prod(factor_sizes) / np.cumprod(factor_sizes)
        indices = np.array(np.dot(all_factors, factor_bases), dtype=np.int64)

        return np.expand_dims(images[indices].astype(np.float32), axis=3)

    