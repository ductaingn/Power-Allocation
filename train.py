import random
import numpy as np
import torch
from typing import Optional, Callable
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from environment.gym_env.wireless_env_sacpa import WirelessEnvironmentSACPA
from environment.gym_env.wireless_env_sacpf import WirelessEnvironmentSACPF
from environment.gym_env.wireless_env_raql import WirelessEnvironmentRAQL
from environment.gym_env.wireless_env_random import WirelessEnvironmentRandom
from architectures import CustomFeatureExtractor
import yaml
from datetime import datetime
import wandb
from utils.logger import WandbLoggingCallback, custom_callback

def exponential_decay(initial_value: float, decay_rate: float) -> Callable[[float], float]:
    """
    Exponentially decays the learning rate

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return func

def make_env(config, seed, algorithm):
    def _init():
        if algorithm == "SACPA":
            return WirelessEnvironmentSACPA(**config, seed=seed)
        elif algorithm == "SACPF":
            return WirelessEnvironmentSACPF(**config, seed=seed)
        elif algorithm == "RAQL":
            return WirelessEnvironmentRAQL(**config, seed=seed)
        elif algorithm == "Random":
            return WirelessEnvironmentRandom(**config, seed=seed)
    return _init

class Trainer:
    def __init__(self, train_configs:dict):
        self.train_configs = train_configs
        self.num_envs = train_configs['num_envs']
        self.num_episodes_per_env = train_configs['num_episodes_per_env']
        self.env_config = train_configs['env_config']
        self.seed = train_configs.get('seed', 1)

        if torch.cuda.is_available():
            self.device = torch.cuda.get_device_name(0)
        else:
            self.device = 'cpu'                

    def train(self, run_name:Optional[str]=None):
        max_steps = self.env_config['max_steps']
        algorithm = self.env_config['algorithm']            

        envs = DummyVecEnv([make_env(config=self.env_config, seed=self.seed+i, algorithm=algorithm) for i in range(self.num_envs)])

        policy_kwargs = dict(
            features_extractor_class = CustomFeatureExtractor,
            features_extractor_kwargs = dict(
                state_dim=8,
                latent_dim=256,
                num_devices=self.env_config['num_devices'],
            )
        )

        time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

        wandb_config = self.train_configs.copy()

        wandb.init(project=self.train_configs['wandb']['project'], config=wandb_config, name=run_name)

        logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])

        model = SAC('MlpPolicy', envs, policy_kwargs=policy_kwargs, verbose=1, seed=self.seed, device=self.device, ent_coef="auto", gamma=0.99, tau=0.005, learning_starts=100, learning_rate=get_linear_fn(0.01, 0, 1))
        model.set_logger(logger)

        print(f"Training {algorithm} with {self.num_envs} environments for {self.num_episodes_per_env} episodes each.")
        print(f"Device: {self.device}, Seed: {self.seed}, Max Steps: {max_steps}")

        if algorithm == "Random":
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
            model.save(f'sb3_trained_weight/{algorithm}/{time_now}')
        elif algorithm == "RAQL":
            model = SAC('MlpPolicy', envs, verbose=1, seed=self.seed, device=self.device, ent_coef="auto", gamma=0.99, tau=0.005, learning_starts=100, learning_rate=get_linear_fn(0.01, 0, 1))
            model.set_logger(logger)
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
        else:
            model.learn(total_timesteps=max_steps*self.num_envs*self.num_episodes_per_env, progress_bar=True, log_interval=1, callback=WandbLoggingCallback(logger), learning_start=100)
            model.save(f'sb3_trained_weight/{algorithm}/{time_now}')
        envs.close()

        wandb.finish(exit_code=0)


if __name__ == "__main__":
    train_configs:dict = yaml.safe_load(open("train_config.yaml"))
    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(train_configs=train_configs)
    trainer.train()