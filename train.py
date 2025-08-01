import random
import numpy as np
import torch
import pandas as pd
from typing import Optional, Callable, Union
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from environment.gym_env.wireless_env_sacpa import WirelessEnvironmentSACPA
from environment.gym_env.wireless_env_sacpf import WirelessEnvironmentSACPF
from environment.gym_env.wireless_env_raql import WirelessEnvironmentRAQL
from environment.gym_env.wireless_env_random import WirelessEnvironmentRandom
from architectures import CustomFeatureExtractor
import yaml
from datetime import datetime
import wandb
from utils.logger import WandbLoggingCallback, custom_callback, get_log_from_wandb
from utils.process_results import get_result_table

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
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")
    return _init

class Trainer:
    def __init__(self, train_configs:dict):
        self.train_configs = train_configs
        self.num_envs = train_configs['num_envs']
        self.num_episodes_per_env = train_configs['num_episodes_per_env']
        self.env_config = train_configs['env_config']
        self.seed = train_configs.get('seed', 1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"                

    def train(self, run_name:Optional[str]=None, tune:bool=False) -> Union[None, pd.DataFrame]:
        """
        Train the model with the given configurations.

        :param run_name: Name of the run for logging purposes.
        :param tune: If True, will tune hyperparameters using Optuna.

        :return: None
        """
        max_steps = self.env_config['max_steps']
        algorithm = self.env_config['algorithm']            

        envs = DummyVecEnv([
            lambda i=i: Monitor(make_env(config=self.env_config, seed=self.seed+i, algorithm=algorithm)())
            for i in range(self.num_envs)
        ])

        policy_kwargs = dict(
            features_extractor_class = CustomFeatureExtractor,
            features_extractor_kwargs = dict(
                state_dim=8,
                latent_dim=256,
                num_devices=self.env_config['num_devices'],
            )
        )

        sac_hyperparams:dict = self.train_configs.get('sac_hyperparams', {})

        time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

        wandb_run = wandb.init(
            project=self.train_configs['wandb']['project'], 
            config=self.train_configs.copy(), 
            name=run_name,
        )

        logger = configure(
            folder=f"training_log/{time_now}", 
            format_strings=["stdout","csv"]
        )

        model = SAC(
            'MlpPolicy', 
            envs, 
            policy_kwargs = policy_kwargs, 
            verbose = 1, 
            seed = self.seed, 
            device = self.device, 
            ent_coef = "auto", 
            gamma = sac_hyperparams.get("gamma", 0.99), 
            tau = sac_hyperparams.get("tau", 0.005), 
            learning_rate = sac_hyperparams.get("learning_rate",get_linear_fn(0.01, 0, 1)),
            learning_starts = 100, 
        )

        model.set_logger(logger)

        print(f"Training {algorithm} with {self.num_envs} environments for {self.num_episodes_per_env} episodes each.")
        print(f"Device: {self.device}, Seed: {self.seed}, Max Steps: {max_steps}")

        if algorithm == "Random":
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
        elif algorithm == "RAQL":
            model = SAC('MlpPolicy', envs)
            evaluate_policy(model, envs, n_eval_episodes=1, callback=custom_callback)
        else:
            model.learn(
                total_timesteps=max_steps*self.num_envs*self.num_episodes_per_env,
                progress_bar=True, 
                log_interval=1, 
                callback=WandbLoggingCallback(logger), 
            )
            model.save(f'sb3_trained_weight/{algorithm}/{time_now}')
            
        envs.close()

        wandb.finish(exit_code=0)

        if tune:
            results = get_result_table(
                history_dfs=[get_log_from_wandb(id=wandb_run.id, return_run=False)],
                num_devices= self.env_config['num_devices']
            )
            print("Tuning completed successfully.")

            return results
        else:
            print("Training completed successfully.")
            return None

if __name__ == "__main__":
    train_configs:dict = yaml.safe_load(open("train_config.yaml"))
    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(train_configs=train_configs)
    trainer.train()