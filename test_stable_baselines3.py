from stable_baselines3 import SAC, PPO, TD3
from vec_env import WirelessEnvironment
import yaml
import gymnasium as gym
import numpy as np
import traceback
import os
os.environ["QT_QPA_PLATFORM"] = "offscreens"

if __name__ == "__main__":
    train_configs = yaml.safe_load(open("train_config.yaml"))
    env = WirelessEnvironment(**train_configs)

    model = TD3.load('/home/nguyen/Projects/Group ICN/Power Allocation/Source Code/trained_weights/td3_model/SB3-2025-04-06-16-42-18/TD3_model_7500_steps.zip')
    model.set_parameters('/home/nguyen/Projects/Group ICN/Power Allocation/Source Code/trained_weights/td3_model/SB3-2025-04-06-16-42-18/TD3_model_7500_steps.zip')
    model.load_replay_buffer('/home/nguyen/Projects/Group ICN/Power Allocation/Source Code/trained_weights/td3_model/SB3-2025-04-06-16-42-18/TD3_model_7500_steps.zip')
    state, info = env.reset()
    while True:
        action, _states = model.predict(state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()