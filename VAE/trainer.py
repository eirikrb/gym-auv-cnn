import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Trainer():
    """Trainer class for training VAE models"""
    def __init__(self,
                 model:nn.Module,
                 epochs:int,
                 learning_rate:float,
                 batch_size:int,
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 beta:float=1
                 ) -> None:
    
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.beta = beta
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.training_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}
        self.validation_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}

    def reconstruction_loss_weights(self, x, k): # Not in use as per now. Idea: Scales losses by how close obstacle is to vehicle, i.e. how close values in the observation are to 1 (collision)
        ones = torch.ones_like(x)
        closeness_to_one = ones/(ones + torch.exp(-5*(x-0.5))) # sigmoid function on interval [0,1]. -5 sets the steepness of the curve.
        scaling = k * closeness_to_one # scaling of the non-linear closeness
        return scaling 

    def loss_function(self, x_hat, x, mu, log_var, beta, N, M):
        """Calculates loss function for VAE, returns (total loss, reconstruction loss, KL divergence loss)"""
        BCE_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')#, weight=BCE_weights) # Reconstruction loss
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
        # N and M stuff is deprecated, remove later
        N = 1
        M = 1
        return (N/M)*(BCE_loss + beta*KLD_loss), (N/M)*BCE_loss, (N/M)*KLD_loss
    
    def train_epoch(self):
        """Trains model for one epoch"""
        self.model.train()
        tot_train_loss = 0.0
        bce_train_loss = 0.0
        kl_train_loss = 0.0

        for x_batch in self.dataloader_train:
            x_batch = x_batch.to(self.device)
            x_hat, mu, log_var = self.model(x_batch)
            loss, bce_loss, kl_loss = self.loss_function(x_hat, x_batch, mu, log_var, self.beta, N=len(self.dataloader_train.dataset), M=len(x_batch))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tot_train_loss += loss.item()
            bce_train_loss += bce_loss.item()
            kl_train_loss += kl_loss.item()
        
        avg_tot_train_loss = tot_train_loss/len(self.dataloader_train.dataset)
        avg_bce_train_loss = bce_train_loss/len(self.dataloader_train.dataset)
        avg_kl_train_loss = kl_train_loss/len(self.dataloader_train.dataset)

        return avg_tot_train_loss, avg_bce_train_loss, avg_kl_train_loss
    
    def validation(self):
        """Calculates validation loss"""
        self.model.eval()

        tot_val_loss = 0.0
        bce_val_loss = 0.0
        kl_val_loss = 0.0

        with torch.no_grad():
            for x in self.dataloader_val:
                x = x.to(self.device)
                x_hat, mu, log_var = self.model(x)
                loss, bce_loss, kl_loss = self.loss_function(x_hat, x, mu, log_var, self.beta, N=len(self.dataloader_train.dataset), M=len(x))
                tot_val_loss += loss.item()
                bce_val_loss += bce_loss.item()
                kl_val_loss += kl_loss.item()

        
        avg_tot_val_loss = tot_val_loss/len(self.dataloader_val.dataset)
        avg_bce_val_loss = bce_val_loss/len(self.dataloader_val.dataset)
        avg_kl_val_loss = kl_val_loss/len(self.dataloader_val.dataset)

        return avg_tot_val_loss, avg_bce_val_loss, avg_kl_val_loss


    def train(self):
        """Main training loop. Trains model for self.epochs epochs"""
        for e in range(self.epochs):

            # Train and validate for given epoch
            train_loss, train_loss_bce, train_loss_kl = self.train_epoch()
            val_loss, val_loss_bce, val_loss_kl = self.validation()
            # Training losses
            self.training_loss['Total loss'].append(train_loss)
            self.training_loss['Reconstruction loss'].append(train_loss_bce)
            self.training_loss['KL divergence loss'].append(train_loss_kl)
            # Validation losses
            self.validation_loss['Total loss'].append(val_loss)
            self.validation_loss['Reconstruction loss'].append(val_loss_bce)
            self.validation_loss['KL divergence loss'].append(val_loss_kl)

            print(f'\nEpoch: {e+1}/{self.epochs} \t|\t Train loss: {train_loss:.3f} \t|\t Val loss: {val_loss:.3f}')




# TODO: Implement trainer with infoVAE loss




###########################################################################################################
class Trainer_old():
    """Trainer class for training VAE models, deprecated as per December 2023"""
    def __init__(self,
                 model:nn.Module,
                 epochs:int,
                 learning_rate:float,
                 batch_size:int,
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 beta:float=1
                 ) -> None:
    
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.beta = beta
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.training_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}
        self.validation_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}
    

    def loss_function(self, x_hat, x, mu, log_var, beta):
        """Calculates loss function for VAE, returns (total loss, reconstruction loss, KL divergence loss)"""
        BCE_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') # Reconstruction loss
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # KL divergence loss
        return BCE_loss + beta*KLD_loss, BCE_loss, KLD_loss
    
    def train_epoch(self):
        """Trains model for one epoch"""
        self.model.train()

        tot_train_loss = 0.0
        bce_train_loss = 0.0
        kl_train_loss = 0.0

        for x_batch in self.dataloader_train:
            x_batch = x_batch.to(self.device)
            x_hat, mu, log_var = self.model(x_batch)
            loss, bce_loss, kl_loss = self.loss_function(x_hat, x_batch, mu, log_var, self.beta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tot_train_loss += loss.item()
            bce_train_loss += bce_loss.item()
            kl_train_loss += kl_loss.item()

        avg_tot_train_loss = tot_train_loss/len(self.dataloader_train.dataset)
        avg_bce_train_loss = bce_train_loss/len(self.dataloader_train.dataset)
        avg_kl_train_loss = kl_train_loss/len(self.dataloader_train.dataset)

        return avg_tot_train_loss, avg_bce_train_loss, avg_kl_train_loss
    
    def validation(self):
        """Calculates validation loss"""
        self.model.eval()

        tot_val_loss = 0.0
        bce_val_loss = 0.0
        kl_val_loss = 0.0

        with torch.no_grad():
            for x in self.dataloader_val:
                x = x.to(self.device)
                x_hat, mu, log_var = self.model(x)
                loss, bce_loss, kl_loss = self.loss_function(x_hat, x, mu, log_var, self.beta)
                tot_val_loss += loss.item()
                bce_val_loss += bce_loss.item()
                kl_val_loss += kl_loss.item()

        avg_tot_val_loss = tot_val_loss/len(self.dataloader_val.dataset)
        avg_bce_val_loss = bce_val_loss/len(self.dataloader_val.dataset)
        avg_kl_val_loss = kl_val_loss/len(self.dataloader_val.dataset)  

        return avg_tot_val_loss, avg_bce_val_loss, avg_kl_val_loss


    def train(self):
        """Main training loop. Trains model for self.epochs epochs"""
        for e in range(self.epochs):

            # Train and validate for given epoch
            train_loss, train_loss_bce, train_loss_kl = self.train_epoch()
            val_loss, val_loss_bce, val_loss_kl = self.validation()
            # Training losses
            self.training_loss['Total loss'].append(train_loss)
            self.training_loss['Reconstruction loss'].append(train_loss_bce)
            self.training_loss['KL divergence loss'].append(train_loss_kl)
            # Validation losses
            self.validation_loss['Total loss'].append(val_loss)
            self.validation_loss['Reconstruction loss'].append(val_loss_bce)
            self.validation_loss['KL divergence loss'].append(val_loss_kl)

            print(f'\nEpoch: {e+1}/{self.epochs} \t|\t Train loss: {train_loss:.3f} \t|\t Val loss: {val_loss:.3f}')
