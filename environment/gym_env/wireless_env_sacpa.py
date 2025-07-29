from environment.gym_env.Environment import W_SUB, W_MW, SIGMA_SQR
from .wireless_env_base import WirelessEnvironmentBase, ln2
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torch.nn.functional import softmax
import numpy as np
from typing import Optional
import attrs

class WirelessEnvironmentSACPA(WirelessEnvironmentBase):
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

        if self.algorithm == "SACPA":
            num_send_packet, power = self.compute_number_send_packet_and_power(policy_network_output, l_max_estimate)
            allocation = self.allocate(num_send_packet)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        # if self.algorithm == "WaterFilling":
        #     num_send_packet, _ = self.compute_number_send_packet_and_power(policy_network_output, l_max_estimate)
        #     allocation = self.allocate(num_send_packet)
        #     power = self.water_filling(allocation, epsilon=0.1)
        
        return num_send_packet, power, allocation

    def water_filling(self, allocation, epsilon):
        from environment.gym_env.Environment import W_SUB, W_MW, SIGMA_SQR
        power:float
        sum_numerator = 0
        sum_denominator = 0
        h = np.zeros(shape=(self.num_devices, 2))

        for k in range(self.num_devices):
            if allocation[k,0]!=-1:
                sub_channel_index = allocation[k,0]
                h[k,0] = self.compute_h_sub(
                    device_position=self.device_positions[k], 
                    h_tilde=self.h_tilde[self.current_step, 0, k, sub_channel_index])
                sum_numerator += W_SUB*SIGMA_SQR/h[k,0]

                sum_denominator += W_SUB**2*SIGMA_SQR/ln2

            if allocation[k,1]!=-1:
                mW_beam_index = allocation[k,1]
                h[k,1] = self.compute_h_mW(
                    device_position=self.device_positions[k], device_index=k, 
                    h_tilde=self.h_tilde[self.current_step, 1, k, mW_beam_index])
                sum_numerator += W_MW*SIGMA_SQR/h[k,1]

                sum_denominator += W_MW**2*SIGMA_SQR/ln2

        upperbound = (self.P_sum + sum_numerator)/(sum_denominator)
        lowerbound = 0


        power:np.ndarray = np.zeros(shape=(self.num_devices, 2))
        while True:
            alpha_ = (upperbound+lowerbound)/2
            for k in range(self.num_devices):
                if allocation[k,0]!=-1:
                    power[k,0] = max(0, W_SUB*SIGMA_SQR*(alpha_*W_SUB/ln2 - 1/h[k,0]))
                if allocation[k,1]!=-1:
                    power[k,1] = max(0, W_MW*SIGMA_SQR*(alpha_*W_MW/ln2 - 1/h[k,1]))

            if self.P_sum - power.sum() < epsilon:
                break

            print("power sum")
            print(power.sum())
            print("power sub")
            print(power[:,0])
            print("power mw")
            print(power[:,1])
            print(f'Upperbound: {upperbound}\nLowerbound: {lowerbound}')
            import time
            time.sleep(1)

            if self.P_sum > power.sum():
                upperbound = alpha_
            else:
                lowerbound = alpha_

        return power/self.P_sum
    
    def compute_number_send_packet_and_power(self, policy_network_output:torch.Tensor, l_max_estimate:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        power_start_index = 2*self.num_devices
        interface_score = policy_network_output[:power_start_index].reshape(self.num_devices, 2)
        interface_score = torch.softmax(torch.tensor(interface_score), dim=1).numpy()

        number_of_send_packet = np.minimum(np.minimum(
            interface_score*self.L_max,
            l_max_estimate,
        ).astype(int), self.L_max)

        power = policy_network_output[power_start_index:]
        power = torch.softmax(torch.tensor(power), dim=-1).numpy()
        power = power.reshape(self.num_devices, 2)

        for k in range(self.num_devices):
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] == 0: # Force to send at least one packet on more reliable channel
                if self.packet_loss_rate[k,0] < self.packet_loss_rate[k,1]:
                    number_of_send_packet[k,0] = 1
                else:
                    number_of_send_packet[k,1] = 1
            
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] > self.L_max:
                # If the number of packets to send exceeds the maximum number of packets that can be sent
                # then send on both channels by the proportion of the packet success rate
                if np.sum(self.packet_loss_rate[k]) == 0:
                    psr_proportion = 0.5
                else:
                    psr_proportion = 1 - self.packet_loss_rate[k,0]/np.sum(self.packet_loss_rate[k])
                number_of_send_packet[k,0] = np.floor(psr_proportion*self.L_max)
                number_of_send_packet[k,1] = self.L_max - number_of_send_packet[k,0]
            
            # Send the remaining power to the other channel
            if number_of_send_packet[k,0] == 0:
                power[k,1] += power[k,0]
                power[k,0] = 0
            if number_of_send_packet[k,1] == 0:
                power[k,0] += power[k,1]
                power[k,1] = 0

        return number_of_send_packet, power

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
    id='WirelessEnvironmentSACPA-v1',
    entry_point='WirelessEnvironmentSACPA',
)