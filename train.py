import random
import numpy as np
import torch
from stable_baselines3 import SAC, PPO, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from vec_env import WirelessEnvironment
from architectures import CustomFeatureExtractor
import yaml
from datetime import datetime
import wandb
from helper import WandbLoggingCallback

def make_env(config, seed):
    def _init():
        return WirelessEnvironment(**config, seed=seed)
    return _init

if __name__ == "__main__":
    train_configs:dict = yaml.safe_load(open("train_config.yaml"))
    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = train_configs['num_envs']
    num_episodes_per_env = train_configs['num_episodes_per_env']
    env_config = train_configs['env_config']
    max_steps = env_config['max_steps']

    envs = SubprocVecEnv([make_env(config=env_config, seed=seed+i) for i in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs = dict(
            state_dim=8,
            latent_dim=256,
            num_devices=10,
        )
    )

    time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

    wandb_config = train_configs.copy()
    wandb_config["algorithm"] = env_config["algorithm"]

    wandb.init(project='PowerAllocation', config=wandb_config)

    logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])

    model = SAC('MlpPolicy', envs, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
    # model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.set_logger(logger)

    model.learn(total_timesteps=max_steps*num_envs*num_episodes_per_env, progress_bar=True, log_interval=1, callback=WandbLoggingCallback(logger))
    model.save(f'sb3_trained_weight/sac_model/{time_now}')
    envs.close()