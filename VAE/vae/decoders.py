import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BaseDecoder(nn.Module):
    """Base class for convolutional decoder"""
    def __init__(self, latent_dims:int=12, n_sensors:int=180) -> None:
        super(BaseDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.n_sensors = n_sensors

    def circular_tconv_1d(self, deconv_block:nn.ConvTranspose1d, z:torch.Tensor, kernel_size:int, stride:int):
        """
        Circular transposed convolution in one dimension given a deconvolution block with some kernel size and stride
        Following the Tensorflow 2D code from: https://www.tu-chemnitz.de/etit/proaut/en/research/rsrc/ccnn/code/ccnn_layers.py 
        """
        pad_width = int(
            0.5 + (kernel_size - 1.) / (2. * stride))  # ceil
        crop = pad_width * stride

        # concatenate the first pad_width (left) columns to the last (pad_width) right columns, and
        # concatenate the last pad_width (right) columns to the first (pad_width) left columns
        pad_left = z[:, :, :pad_width]
        pad_right = z[:, :, -pad_width:]
        z_concat = torch.cat((pad_left, z, pad_right), dim=2)

        # Perform circular convolution with regular zero-padding ('same' in tensorflow code):
        z_t_conv = deconv_block(z_concat)

        # Remove the first crop left columns and the last pw/2 right columns and return
        z_t_conv_cropped = z_t_conv[:, :, crop:-crop]
        return z_t_conv_cropped
    
    @abstractmethod
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass

    def save(self, path:str) -> None:
        """Saves model to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        """Loads model from path"""
        self.load_state_dict(torch.load(path))


class Decoder_conv_shallow(BaseDecoder):
    """ 
    Shallow convolutional decoder
    One convolutional layer and one fully connected layer
    Zero padding
    """
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 kernel_overlap = 0.25,
                 kernel_size=None):  
        super().__init__()
        self.latent_dims = latent_dims
        self.in_channels = 1 
        self.out_channels = 1
        if kernel_size is None:
            self.kernel_size = round(n_sensors * kernel_overlap)  # 45
            self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        else:
            self.kernel_size = kernel_size

        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding


        len_flat = int((n_sensors - self.kernel_size + 2*self.padding) // self.stride + 1) # D_out = ((D_in - kernel_size + 2*padding) / stride) + 1 
        #len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dims, len_flat), 
            nn.ReLU()
        )

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                padding_mode='zeros'
            ),
            nn.Sigmoid() # alle har bare sigmoid her, idk why
        )#'circular' is not supported for convtranspose1d :(

    def forward(self, z):
        z = self.linear(z)
        x_hat = self.decoder_layer1(z)
        return x_hat


class Decoder_circular_conv_shallow2(BaseDecoder):
    """
    Shallow convolutional decoder
    One convolutional layer and one fully connected layer
    Circular padding (https://www.tu-chemnitz.de/etit/proaut/publications/schubert19_IV.pdf chapter III section B)
    """
    def __init__(self,
                 latent_dims: int = 12,
                 n_sensors: int = 180,
                 kernel_overlap=0.25,
                 kernel_size=None):
        super().__init__()
        self.latent_dims = latent_dims
        self.in_channels = 1
        self.out_channels = 1
        if kernel_size is None:
            self.kernel_size = round(n_sensors * kernel_overlap)  # 45
            self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        else:
            self.kernel_size = kernel_size

        # self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding

        len_flat = int((n_sensors - self.kernel_size + 2*self.padding) // self.stride + 1) # D_out = ((D_in - kernel_size + 2*padding) / stride) + 1 
        #len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dims, len_flat),
            nn.ReLU()
        )

        # No nn.sequentioal bc. of circular padding in between layers
        self.deconv_block = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding, # added to look like the code in the paper
        )

        self.sigmoid = nn.Sigmoid()
   

    def forward(self, z):
        z = self.linear(z)
        z_t_conv = super().circular_tconv_1d(deconv_block=self.deconv_block, z=z, kernel_size=self.kernel_size, stride=self.stride)
        x_hat = self.sigmoid(z_t_conv)
        return x_hat



class Decoder_circular_conv_deep(BaseDecoder):
    """
    Deep convolutional decoder
    Three convolutional layers and one fully connected layer
    Circular padding
    """
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 output_channels:list=[1,2,3], 
                 kernel_size:int=45,):  
        super().__init__()

        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.latent_dims = latent_dims

        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(
            nn.Linear(self.latent_dims, len_flat),
            nn.ReLU()
        )        

        self.deconv1 = nn.ConvTranspose1d(
            in_channels  = self.output_channels[0],
            out_channels = self.output_channels[1],
            kernel_size  = 3,
            stride       = 1,
            padding      = 1
        )

        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose1d(
            in_channels  = self.output_channels[1],
            out_channels = self.output_channels[2],
            kernel_size  = 3,
            stride       = 1,
            padding      = 1
        )

        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose1d(
            in_channels  = self.output_channels[2],
            out_channels = 1,
            kernel_size  = 45,
            stride       = 15,
            padding      = 15
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z1 = self.linear(z)
        z2 = super().circular_tconv_1d(deconv_block=self.deconv1, z=z1, kernel_size=3, stride=1)
        z3 = self.relu1(z2)
        z4 = super().circular_tconv_1d(deconv_block=self.deconv2, z=z3, kernel_size=3, stride=1)
        z5 = self.relu2(z4)
        z6 = super().circular_tconv_1d(deconv_block=self.deconv3, z=z5, kernel_size=45, stride=15)
        x_hat = self.sigmoid(z6)
        return x_hat



class Decoder_conv_deep(BaseDecoder):
    """
    Deep convolutional decoder
    Three convolutional layers and one fully connected layer
    Zero padding
    """
    def __init__(self, 
                 latent_dims:int=12,
                 n_sensors:int=180, 
                 output_channels:list=[1,2,3], 
                 kernel_size:int=45,):  
        super().__init__()

        self.n_sensors = n_sensors
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.latent_dims = latent_dims

        len_flat = 12 # bc. of combination of input dim, kernel, stride and padding. TODO: Automate.

        self.linear = nn.Sequential(nn.Linear(self.latent_dims, len_flat), nn.ReLU())
        
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[0],
                out_channels = self.output_channels[1],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[1],
                out_channels = self.output_channels[2],
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                padding_mode = 'zeros'
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels  = self.output_channels[2],
                out_channels = 1,
                kernel_size  = 45,
                stride       = 15,
                padding      = 15,
                padding_mode = 'zeros'
            ),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.linear(z)
        x_hat = self.deconv_block(z)
        return x_hat