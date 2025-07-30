from environment.gym_env.Environment import W_SUB, W_MW, SIGMA_SQR
from .wireless_env_base import WirelessEnvironmentBase, ln2
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torch.nn.functional import softmax
import numpy as np
from typing import Optional
import attrs

class WirelessEnvironmentRandom(WirelessEnvironmentBase):
    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self._observation_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices)), # Quality of Service Satisfaction of each device on Sub6GHz
                np.zeros(shape=(self.num_devices)), # Quality of Service Satisfaction of each device on mmWave,
                np.zeros(shape=(self.num_devices)), # Number of received packets of each device on Sub6GHz of previous time step
                np.zeros(shape=(self.num_devices)), # Number of received packets of each device on mmWave of previous time step
                np.zeros(shape=(self.num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.zeros(shape=(self.num_devices)), # Average Rate of each device on mmWave of previous time step
                np.zeros(shape=(self.num_devices)), # Power of each device on Sub6GHz on previous time step
                np.zeros(shape=(self.num_devices)), # Power of each device on mmWave on previous time step
            ]).transpose().flatten(),
            high=np.array([
                np.ones(shape=(self.num_devices)), # Quality of Service Satisfaction of each device on Sub6GHz
                np.ones(shape=(self.num_devices)), # Quality of Service Satisfaction of each device on mmWave,
                np.full(shape=(self.num_devices), fill_value=self.L_max), # Number of received packets of each device on Sub6GHz of previous time step
                np.full(shape=(self.num_devices), fill_value=self.L_max), # Number of received packets of each device on mmWave of previous time step,
                np.ones(shape=(self.num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.ones(shape=(self.num_devices)), # Average Rate of each device on mmWave of previous time step,
                np.ones(shape=(self.num_devices)), # Power of each device on Sub6GHz of previous time step
                np.ones(shape=(self.num_devices)), # Power of each device on mmWave of previous time step,
            ]).transpose().flatten()
        )

        self._action_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.zeros(shape=(self.num_devices)), # Number of packets to send of each device on mmWave,
                np.zeros(shape=(self.num_devices)), # Power of each device on Sub6GHz,
                np.zeros(shape=(self.num_devices)), # Power of each device on mmWave,
            ]).flatten(),
            high=np.array([
                np.ones(shape=(self.num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.ones(shape=(self.num_devices)), # Number of packets to send of each device on mmWave,
                np.ones(shape=(self.num_devices)), # Power of each device on Sub6GHz,
                np.ones(shape=(self.num_devices)), # Power of each device on mmWave,
            ]).flatten()
        )

        self._state = np.zeros(shape=(self.num_devices, 8))
        self._action = np.zeros(shape=(self.num_devices, 4))

        self._estimated_ideal_power = np.zeros(shape=(self.num_devices, 2))
    
    @property
    def estimated_ideal_power(self) -> np.ndarray:
        return self._estimated_ideal_power
    
    def get_state(self, num_received_packet:np.ndarray, power:np.ndarray) -> np.ndarray:
        _state = np.zeros(shape=(self.num_devices, 8))
        # QoS satisfaction
        _state[:, 0] = (self.packet_loss_rate[:, 0] <= self.qos_threshold).astype(float)
        _state[:, 1] = (self.packet_loss_rate[:, 1] <= self.qos_threshold).astype(float)
        _state[:, 2] = num_received_packet[:, 0].copy()
        _state[:, 3] = num_received_packet[:, 1].copy()
        _state[:, 4] = self.average_rate[:, 0]/self.maximum_rate[0]
        _state[:, 5] = self.average_rate[:, 1]/self.maximum_rate[1]
        _state[:, 6] = power[:, 0].copy()*10.0 # Scale up
        _state[:, 7] = power[:, 1].copy()*10.0
        
        return _state
    
    def get_action(self, policy_network_output:torch.tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        l_max_estimate = self.estimate_l_max()

        if self.algorithm == "Random":
            num_send_packet, power = self.compute_number_send_packet_and_power()
            allocation = self.allocate(num_send_packet)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")        

        return num_send_packet, power, allocation

    def compute_number_send_packet_and_power(self) -> tuple[np.ndarray, np.ndarray]:
        num_send_packet:np.ndarray = np.random.randint(0, self.L_max, (self.num_devices,2))
        power = torch.softmax(torch.tensor(np.random.rand(self.num_devices*2)), dim=-1).reshape(self.num_devices, 2).numpy()

        for k in range(self.num_devices):
            if num_send_packet[k].sum() == 0:
                if np.random.rand()>0.5:
                    num_send_packet[k,0] = 1
                    power[k,1] = 0 # For analyzing purpose
                else:
                    num_send_packet[k,1] = 1
                    power[k,0] = 0
            if num_send_packet[k].sum() > self.L_max:
                proportion = num_send_packet[k,0]/num_send_packet[k].sum()
                num_send_packet[k,0] = np.floor(proportion*self.L_max)
                num_send_packet[k,1] = self.L_max-num_send_packet[k,0]
        
        return num_send_packet, power

    def step(self, policy_network_output):
        state = None
        reward = None
        terminated = False
        truncated = False

        num_send_packet, power, allocation = self.get_action(policy_network_output)
        num_received_packet = self.get_feedback(allocation, num_send_packet, power)
        self.average_rate = self.compute_average_rate()
        self.channel_power_gain = self.estimate_average_channel_power(num_received_packet, power, allocation)

        reward_qos, reward_power, reward = self.get_reward(num_send_packet, num_received_packet, power)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("Reward is NaN or Inf")

        state = self.get_state(num_received_packet, power)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError("State contains NaN or Inf values")
        observation = state.flatten()

        info = self.get_info(
            reward=reward,
            reward_qos=reward_qos,
            reward_power=reward_power,
            power=power,
            num_sent_packet=num_send_packet,
            num_received_packet=num_received_packet
        )

        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._state = np.zeros(shape=(self.num_devices, 8))
        self._action = np.zeros(shape=(self.num_devices, 4))

        return super().reset(seed, options)
    
    def get_reward(self, num_sent_packet, num_received_packet, power):
        def estimate_ideal_power(num_send_packet, channel_power, W):
            if channel_power==0:
                return self.P_sum
            
            ideal_power = (2**((num_send_packet*self.D)/(W*self.T)) - 1) * \
                        W*SIGMA_SQR/channel_power
            return min(ideal_power, self.P_sum)
        
        reward_qos = 0
        reward_power = 0
        target_power = []
        predicted_power = []

        for k in range(self.num_devices):
            power_sub, power_mw = power[k, 0], power[k, 1] # Unit: percentage
            qos_satisfaction = self.state[k, 0], self.state[k, 1]
            
            reward_qos += (num_received_packet[k,0] + num_received_packet[k,1])/(num_sent_packet[k,0] + num_sent_packet[k,1]) - (1-qos_satisfaction[0]) - (1-qos_satisfaction[1])

            if num_sent_packet[k,0] > 0:
                self.estimated_ideal_power[k,0] = estimate_ideal_power(num_sent_packet[k,0], self.channel_power_gain[k,0], W_SUB)
                target_power.append(self.estimated_ideal_power[k,0])
                predicted_power.append(power_sub)

            if num_sent_packet[k,1] > 0:
                self.estimated_ideal_power[k,1] = estimate_ideal_power(num_sent_packet[k,1], self.channel_power_gain[k,1], W_MW)
                target_power.append(self.estimated_ideal_power[k,1])
                predicted_power.append(power_mw)

        target_power = torch.tensor(target_power)
        predicted_power = torch.tensor(predicted_power)

        target_power = softmax(target_power, dim=-1)
        reward_power = -self.num_devices*(target_power*(target_power.log()-predicted_power.log())).sum()
        reward_qos = ((self.current_step-1)*self.reward_qos + reward_qos)/self.current_step

        self.reward_qos = reward_qos
        self.instance_reward = self.reward_coef['reward_qos']*reward_qos + self.reward_coef['reward_power']*reward_power
        
        return reward_qos, reward_power, self.instance_reward
    

register(
    id='WirelessEnvironmentRandom-v1',
    entry_point='WirelessEnvironmentRandom',
)