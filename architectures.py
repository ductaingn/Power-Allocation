from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class BackBone(nn.Module):
    def __init__(self, state_dim, latent_dim, num_devices, *args, **kwargs):
        super(BackBone, self).__init__(*args, **kwargs)
        self.num_devices = num_devices
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(state_dim, 256)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, batch_first=True), num_layers=1)
        self.project = nn.Linear(256*self.num_devices, latent_dim)

    def forward(self, state:torch.Tensor)->torch.Tensor:
        batch_size = state.shape[0]
        out = state.reshape(batch_size, self.num_devices, self.state_dim)
        
        out = self.embed(out)
        out = F.relu(out)

        out = self.transformer(out)
        out = out.reshape(batch_size, -1) # Flatten the output
        out = self.project(out)
        out = F.relu(out)
        
        return out


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.spaces.Box, state_dim:int, latent_dim:int, num_devices:int, features_dim = 256, *args, **kwargs):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.feature_extractor_net = BackBone(state_dim, latent_dim, num_devices)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor_net(observations)
