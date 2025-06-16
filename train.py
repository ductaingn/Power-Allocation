import random
import numpy as np
import torch
from typing import Optional, Callable
from stable_baselines3 import SAC, PPO, TD3, HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from vec_env import WirelessEnvironment
from vec_env_interface_only import WirelessEnvironmentInterfaceOnly
from vec_env_raql import WirelessEnvironmentRiskAverseQLearning
from architectures import CustomFeatureExtractor
import yaml
from datetime import datetime
import wandb
from helper import WandbLoggingCallback, custom_callback

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
        if algorithm == "LearnInterface":
            return WirelessEnvironmentInterfaceOnly(**config, seed=seed)
        elif algorithm == "RAQL":
            return WirelessEnvironmentRiskAverseQLearning(**config, seed=seed)
        else:
            return WirelessEnvironment(**config, seed=seed)
    return _init

class Trainer:
    def __init__(self, train_configs:dict):
        self.train_configs = train_configs
        self.num_envs = train_configs['num_envs']
        self.num_episodes_per_env = train_configs['num_episodes_per_env']
        self.env_config = train_configs['env_config']
        self.seed = train_configs.get('seed', 1)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        

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
        wandb_config["algorithm"] = algorithm

        wandb.init(project='PowerAllocation', config=wandb_config, name=run_name)

        logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])

        model = SAC('MlpPolicy', envs, policy_kwargs=policy_kwargs, verbose=1, seed=self.seed, device=self.device, ent_coef="auto", gamma=0.99, tau=0.005, learning_starts=100, learning_rate=get_linear_fn(0.01, 0, 1))
        model.set_logger(logger)

        if algorithm == "Random":
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
            model.save(f'sb3_trained_weight/sac_model/{time_now}')
        elif algorithm == "RAQL":
            model = SAC('MlpPolicy', envs, verbose=1, seed=self.seed, device=self.device, ent_coef="auto", gamma=0.99, tau=0.005, learning_starts=100, learning_rate=get_linear_fn(0.01, 0, 1))
            model.set_logger(logger)
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
        else:
            model.learn(total_timesteps=max_steps*self.num_envs*self.num_episodes_per_env, progress_bar=True, log_interval=1, callback=WandbLoggingCallback(logger))
            model.save(f'sb3_trained_weight/sac_model/{time_now}')
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