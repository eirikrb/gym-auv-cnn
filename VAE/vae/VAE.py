import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dims:int=12, beta:float=1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.beta = beta
    
    def forward(self, x):
        x = x.to(device)
        mu, sigma, z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma #, z    