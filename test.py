from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from vec_env import WirelessEnvironment
import yaml
import gymnasium as gym
import numpy as np
import os
import wandb
import random
import torch
from helper import WandbLoggingCallback
from datetime import datetime

os.environ["QT_QPA_PLATFORM"] = "offscreens"

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

    env = WirelessEnvironment(**env_config, seed=seed)

    time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

    wandb_config = train_configs.copy()
    wandb_config["algorithm"] = env_config["algorithm"]

    wandb.init(project='PowerAllocation', config=wandb_config)

    model_path = "/home/nguyen/Projects/Group ICN/Power Allocation/Source Code/sb3_trained_weight/td3_model/SB3-2025-04-10-14-29-54.zip"
    model = SAC.load(model_path)
    model.set_parameters(model_path)

    logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])
    model.set_logger(logger)
    
    callback_instance = WandbLoggingCallback(logger)

    def evaluation_callback(locals_, globals_):
        callback_instance.locals = locals_
        callback_instance.globals = globals_
        return callback_instance.on_step()

    evaluate_policy(model, env, 1, callback=evaluation_callback)

    env.close()