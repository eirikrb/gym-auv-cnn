import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Create baseclass for encoder called BaseEncoder
class BaseEncoder(nn.Module):
    """Base class for encoder"""
    def __init__(self, latent_dims:int=12, n_sensors:int=180) -> None:
        super(BaseEncoder, self).__init__()
        self.name = 'base'
        self.latent_dims = latent_dims
        self.n_sensors = n_sensors
    
    def reparameterize(self, mu, log_var, eps_weight=1):
        """Reparameterization trick from VAE paper (Kingma and Welling). Eps weight in [0,1] controls the amount of noise added to the latent space."""
        # Note: log(x²) = 2log(x) -> divide by 2 to get std.dev.
        # Thus, std = exp(log(var)/2) = exp(log(std²)/2) = exp(0.5*log(var))
        std = torch.exp(0.5*log_var)
        epsilon = torch.distributions.Normal(0, eps_weight).sample(mu.shape).to(device) # ~N(0,I)
        z = mu + (epsilon * std)
        return z
    
    @abstractmethod
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass
    
    def save(self, path:str) -> None:
        """Saves model to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        """Loads model from path"""
        self.load_state_dict(torch.load(path))
    

class Encoder_conv_shallow(BaseEncoder):
    """
    Shallow convolutional encoder
    One convolutional layer and one fully connected layer
    Circular padding
    """
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 kernel_overlap:float=0.25,
                 kernel_size=None,
                 eps_weight:float=1):  
        super().__init__()

        self.name = 'conv_shallow'
        self.latent_dims = latent_dims
        self.in_channels = 1

        # Make kernel size adaptable for different sized output from conv-block
        if kernel_size is None:
            self.kernel_size = round(n_sensors * kernel_overlap)  # 45
            self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        else:
            self.kernel_size = kernel_size

        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding
        self.eps_weight = eps_weight
        

        self.encoder_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels  = self.in_channels,
                out_channels = 1,
                kernel_size  = self.kernel_size,
                stride       = self.stride,
                padding      = self.padding,
                padding_mode = 'circular'
            )#,
            #nn.Flatten()
        )
        
        len_flat = int((n_sensors - self.kernel_size + 2*self.padding) // self.stride + 1) # D_out = ((D_in - kernel_size + 2*padding) / stride) + 1 

        #len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) # 
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        mu =  self.fc_mu(x)
        log_var = self.fc_logvar(x)
        z = super().reparameterize(mu, log_var, self.eps_weight)
        return mu, log_var, z # mu and log_var are used in loss function to compute KL divergence loss
    

class Encoder_conv_deep(BaseEncoder):
    """
    Deep convolutional encoder
    Three convolutional layers and one fully connected layer
    Circular padding
    """
    def __init__(self, 
                 n_sensors:int=180, 
                 output_channels:list=[3,2,1], 
                 kernel_size:int=45,
                 latent_dims:int=12,
                 eps_weight:float=1):  
        super().__init__()
        self.name = 'conv_deep'
        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.latent_dims = latent_dims
        self.eps_weight = eps_weight

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels  = 1,
                out_channels = self.output_channels[0],
                kernel_size  = 45,
                stride       = 15,
                padding      = 15,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'circular'
            )#,
            #nn.ReLU()
            #nn.Flatten()
        )
        
        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU())

    def forward(self, x):
        x = x.to(device)
        x = self.conv_block(x)
        mu =  self.fc_mu(x)
        log_var = self.fc_logvar(x)
        z = self.reparameterize(mu, log_var, self.eps_weight)
        return mu, log_var, z