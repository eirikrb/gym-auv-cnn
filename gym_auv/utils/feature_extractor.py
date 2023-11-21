import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ShallowEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (N_sensors x 1)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of latent dimensions
    """

    def __init__(self, 
                 observation_space:gym.spaces.Box, 
                 n_sensors:int=180, 
                 latent_dims:int=12, 
                 kernel_overlap:float=0.25,
                 eps_weight:float=1):
        super(ShallowEncoder, self).__init__(observation_space, features_dim=latent_dims)

        self.latent_dims = latent_dims
        self.in_channels = 1 
        self.kernel_size = round(n_sensors * kernel_overlap)  # 45
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
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
            ),
            nn.Flatten()
        )

        len_flat = 12 

        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 

    def reparameterize(self, mu, log_var, eps_weight):
        std = th.exp(0.5*log_var)
        epsilon = th.distributions.Normal(0, eps_weight).sample(mu.shape) # ~N(0,I)
        z = mu + (epsilon * std)
        return z

    def forward(self, observations:th.Tensor) -> th.Tensor:
        x = self.encoder_layer1(observations)
        mu = self.fc_mu(x)
        #log_var = self.fc_logvar(x)
        #z = self.reparameterize(mu, log_var, self.eps_weight)
        return mu

    def get_features(self, observations:th.Tensor) -> list: # Not in use
        feat = []
        out = observations
        for layer in self.encoder_layer1:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())
        
        for layer in self.fc_mu:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())

        return feat
    
    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.encoder_layer1:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out)

        for layer in self.fc_mu:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().numpy())

        return feat
    
    def load_params(self, path:str) -> None:
        params = th.load(path)
        self.load_state_dict(params)

    def lock_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


class DeepEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (N_sensors x 1)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of latent dimensions
    """

    def __init__(self, 
                 observation_space:gym.spaces.Box, 
                 n_sensors:int=180,
                 output_channels:list=[3,2,1], 
                 kernel_size:int=45,
                 latent_dims:int=12, 
                 kernel_overlap:float=0.25,
                 eps_weight:float=1):
        super(DeepEncoder, self).__init__(observation_space, features_dim=latent_dims)

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
            ),
            nn.Flatten()
        )
        
        len_flat = 12 

        self.fc_mu = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU()) 
        self.fc_logvar = nn.Sequential(nn.Linear(len_flat, self.latent_dims), nn.ReLU())


    def reparameterize(self, mu, log_var, eps_weight):
        std = th.exp(0.5*log_var)
        epsilon = th.distributions.Normal(0, eps_weight).sample(mu.shape) # ~N(0,I)
        z = mu + (epsilon * std)
        return z

    def forward(self, observations:th.Tensor) -> th.Tensor:
        x = self.conv_block(observations)
        mu =  self.fc_mu(x)
        #log_var = self.fc_logvar(x)
        #z = self.reparameterize(mu, log_var, self.eps_weight)
        return mu

    def get_features(self, observations:th.Tensor) -> list: # Not in use (i think)
        feat = []
        out = observations
        for layer in self.encoder_layer1:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())
        
        for layer in self.fc_mu:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())

        return feat
    
    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.encoder_layer1:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out)

        for layer in self.fc_mu:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().numpy())

        return feat
    
    def load_params(self, path:str) -> None:
        params = th.load(path)
        self.load_state_dict(params)

    def lock_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
    


class NavigatioNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(NavigatioNN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        shape = observations.shape
        observations = observations[:,0,:].reshape(shape[0], shape[-1])
        return self.passthrough(observations)


class PerceptionNavigationExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (1, 3, N_sensors)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    #def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 32, kernel_overlap : float = 0.05):
    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim: int = 180, features_dim: int = 12, kernel_overlap: float = 0.25):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionNavigationExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception":
                encoder = ShallowEncoder(subspace, n_sensors=sensor_dim, latent_dims=features_dim, kernel_overlap=kernel_overlap)
                #encoder = DeepEncoder(subspace, n_sensors=sensor_dim, latent_dims=features_dim, kernel_overlap=kernel_overlap)
                # Get params from pre-trained encoder saved in file from path 
                param_path = "/home/eirikrb/Desktop/gym-auv-cnn/gym_auv/utils/pre_trained_encoders/ShallowConvVAE_latent_dims_12_beta_0.1.json"
                #param_path = "/home/eirikrb/Desktop/gym-auv-cnn/gym_auv/utils/pre_trained_encoders/DeepConvVAE_latent_dims_12.json"
                encoder.load_params(param_path)
                encoder.lock_params() # Locks the parameters of the encoder
                extractors[key] = encoder
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "navigation":
                # Pass navigation features straight through to the MlpPolicy.
                extractors[key] = NavigatioNN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

