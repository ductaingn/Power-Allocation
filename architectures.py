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

class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, state_dim, action_dim, num_devices, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        self.mu = nn.Sequential(
            CustomFeatureExtractor(state_dim, action_dim, num_devices),
            nn.Tanh()
        )

class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            q_net = nn.Sequential(...)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

class CustomTD3Policy(TD3Policy):
    """
    Custom TD3 policy with a custom actor and critic.
    """
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

# To register a policy, so you can use a string to create the network
TD3.policy_aliases["CustomTD3Policy"] = CustomTD3Policy
SAC.policy_aliases["CustomSACPolicy"] = CustomSACPolicy