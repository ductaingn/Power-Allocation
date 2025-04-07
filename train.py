from stable_baselines3 import SAC, PPO, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from vec_env import WirelessEnvironment
from architectures import CustomFeatureExtractor
import yaml
from datetime import datetime
if __name__ == "__main__":
    train_configs = yaml.safe_load(open("train_config.yaml"))
    # num_envs = 4
    # envs = SubprocVecEnv([lambda: WirelessEnvironment(**train_configs) for _ in range(num_envs)])

    # policy_kwargs = dict(
    #     features_extractor_class=CustomFeatureExtractor,
    #     features_extractor_kwargs = dict(
    #         state_dim=envs.observation_space.shape[0],
    #         action_dim=envs.action_space.shape[0],
    #         num_devices=10,
    #     )
    # )

    time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

    env = WirelessEnvironment(**train_configs)

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs = dict(
            state_dim=8,
            latent_dim=256,
            num_devices=10,
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=f"trained_weights/td3_model/{time_now}",
        name_prefix="TD3_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    ) 
    logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])

    model = TD3('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    # model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.set_logger(logger)
    env.close()
    model.learn(total_timesteps=10_000, progress_bar=True, log_interval=1, callback=checkpoint_callback)
    model.save(f'sb3_trained_weight/td3_model/{time_now}')