import gymnasium as gym
from gymnasium import Env
from typing import Optional, Literal
import pickle
import numpy as np
import torch
import wandb
from environment.Environment import r_sub as compute_rate_sub, r_mW as compute_rate_mW, G, W_SUB, W_MW, SIGMA_SQR
import random

ln2 = np.log(2)

class WirelessEnvironmentInterfaceOnly(Env):
    def __init__(self, h_tilde_path: str, devices_positions_path: str, L_max: int, T: int, D: int, qos_threshold: float, P_sum, max_steps: int, reward_coef:dict, seed: Optional[int] = None, algorithm: Optional[Literal["LearnInterface"]] = "LearnInterface"):
        super(WirelessEnvironmentInterfaceOnly, self).__init__()
        self.load_h_tilde(h_tilde_path)
        self.load_device_positions(devices_positions_path)
        self.L_max = L_max
        self.T = T
        self.D = D
        self.qos_threshold = qos_threshold
        self.P_sum = P_sum

        self.num_devices = self.device_positions.shape[0]
        self.num_sub_channel = self.h_tilde.shape[-1]
        self.num_beam = self.h_tilde.shape[-1]
        
        self.current_step = 1
        self.max_steps = max_steps
        self.reward_coef = reward_coef

        self.algorithm = algorithm

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        self.seed = seed

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        self.observation_space = gym.spaces.Box(
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

        self.action_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.zeros(shape=(self.num_devices)), # Number of packets to send of each device on mmWave,
            ]).flatten(),
            high=np.array([
                np.ones(shape=(self.num_devices)), # Number of packets to send of each device on Sub6GHz,
                np.ones(shape=(self.num_devices)), # Number of packets to send of each device on mmWave,
            ]).flatten()
        )

        self.state = np.zeros(shape=(self.num_devices, 8))
        self.action = np.zeros(shape=(self.num_devices, 2))
        self.instance_reward = 0.0
        self.reward_qos = 0.0

        self._init_num_send_packet = np.ones(shape=(self.num_devices, 2))
        self._init_power = np.full(shape=(self.num_devices, 2), fill_value=self.P_sum/(self.num_devices*2))
        self._init_allocation = self.allocate(self._init_num_send_packet)
        self._init_rate = self.compute_instant_rate(
            allocation=self._init_allocation,
            power=self._init_power
        )   # shape=(self.num_devices, 2)
        
        self.average_rate = self._init_rate
        self.previous_rate = self._init_rate.copy() # Data rate of previous time step
        self.instant_rate = self._init_rate.copy() # Data rate of current time step (acknowledge after get feedback)
        self.maximum_rate = self.get_maximum_rate() # For normalizing rate to [0,1]

        self.packet_loss_rate = np.zeros((self.num_devices,2)) # Accumulated Packet loss rate of current time step
        self._init_num_received_packet = self.get_feedback(self._init_allocation, self._init_num_send_packet, self._init_power)
        self._init_packet_loss_rate = self.compute_packet_loss_rate(
            self._init_num_received_packet,
            self._init_num_send_packet,
        )
        self.packet_loss_rate = self._init_packet_loss_rate.copy()


    def get_maximum_rate(self):
        maximum_rate = np.zeros(shape=2)
        for k in range(self.num_devices):
            maximum_rate[0] = max(maximum_rate[0], compute_rate_sub(
                h=1.0,
                device_index=k,
                power=self.P_sum
            ))
            maximum_rate[1] = max(maximum_rate[1],compute_rate_mW(
                h=1.0,
                device_index=k,
                power=self.P_sum
            ))

        return maximum_rate
    
    def load_h_tilde(self, h_tilde_path:str):
        with open(h_tilde_path, 'rb') as f:
            h_tilde = np.array(pickle.load(f))
        self.h_tilde = h_tilde        

    def load_device_positions(self, device_positions_path:str):
        with open(device_positions_path, 'rb') as f:
            device_positions = np.array(pickle.load(f))
        self.device_positions = device_positions

    def get_state(self, num_received_packet, power):
        state = np.zeros(shape=(self.num_devices, 8))
        # QoS satisfaction
        state[:, 0] = (self.packet_loss_rate[:, 0] <= self.qos_threshold).astype(float)
        state[:, 1] = (self.packet_loss_rate[:, 1] <= self.qos_threshold).astype(float)
        state[:, 2] = num_received_packet[:, 0].copy()
        state[:, 3] = num_received_packet[:, 1].copy()
        state[:, 4] = self.average_rate[:, 0]/self.maximum_rate[0]
        state[:, 5] = self.average_rate[:, 1]/self.maximum_rate[1]
        state[:, 6] = power[:, 0].copy()
        state[:, 7] = power[:, 1].copy()
        
        self.state = state

        return state
    
    def get_action(self, policy_network_output):
        l_max_estimate = self.estimate_l_max()

        if self.algorithm == "LearnInterface":
            num_send_packet, power = self.compute_number_send_packet_and_power(policy_network_output, l_max_estimate)
            allocation = self.allocate(num_send_packet)

        return num_send_packet, power, allocation
    
    def random_number_send_packet_and_power(self):
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

    def water_filling(self, allocation, epsilon):
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
    
    def estimate_l_max(self):
        # To-do: Might try without estimate_l_max
        l = np.multiply(self.average_rate, self.T/self.D)
        packet_successful_rate = np.ones(shape=(self.num_devices,2)) - self.packet_loss_rate
        l_max_estimate = np.floor(l*packet_successful_rate)

        return l_max_estimate

    def compute_number_send_packet_and_power(self, policy_network_output, l_max_estimate)->tuple[np.ndarray, np.ndarray]:
        interface_score = policy_network_output.reshape(self.num_devices, 2)
        interface_score = torch.softmax(torch.tensor(interface_score), dim=1).numpy()

        number_of_send_packet = np.minimum(np.minimum(
            interface_score*self.L_max,
            l_max_estimate,
        ).astype(int), self.L_max)

        power = np.full(shape=(self.num_devices, 2), fill_value=1.0/(self.num_devices*2))

        for k in range(self.num_devices):
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] == 0: # Force to send at least one packet on Sub6GHz
                number_of_send_packet[k,0] = 1
            
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] > self.L_max:
                # If the number of packets to send exceeds the maximum number of packets that can be sent
                # then send on both channels by the proportion of the packet success rate
                if np.sum(self.packet_loss_rate[k]) == 0:
                    psr_proportion = 0.5
                else:
                    psr_proportion = 1 - self.packet_loss_rate[k,0]/np.sum(self.packet_loss_rate[k])
                number_of_send_packet[k,0] = np.floor(psr_proportion*self.L_max)
                number_of_send_packet[k,1] = self.L_max - number_of_send_packet[k,0]
            
            # For analysing purpose other channel
            if number_of_send_packet[k,0] == 0:
                power[k,0] = 0
            if number_of_send_packet[k,1] == 0:
                power[k,1] = 0

        return number_of_send_packet, power


    def allocate(self, num_send_packet):
        '''
        Allocate subchannel and beam to each device randomly
        num_send_packet: Number of packets to send on each device
        allocation: [subchannel, beam]: shape=(self.num_device, 2)
        '''
        sub = []  # Stores index of subchannel device will allocate
        mW = []  # Stores index of beam device will allocate
        for i in range(self.num_devices):
            sub.append(-1)
            mW.append(-1)

        rand_sub = []
        rand_mW = []
        for i in range(self.num_sub_channel):
            rand_sub.append(i)
        for i in range(self.num_beam):
            rand_mW.append(i)

        for k in range(self.num_devices):
            if (num_send_packet[k,0]>0 and num_send_packet[k,1]==0):
                rand_index = np.random.randint(0,len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
            elif (num_send_packet[k,0]==0 and num_send_packet[k,1]>0):
                rand_index = np.random.randint(0,len(rand_mW))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
            else:
                rand_sub_index = np.random.randint(0,len(rand_sub))
                rand_mW_index = np.random.randint(0,len(rand_mW))

                sub[k] = rand_sub[rand_sub_index]
                mW[k] = rand_mW[rand_mW_index]

                rand_sub.pop(rand_sub_index)
                rand_mW.pop(rand_mW_index)

        allocate = np.array([sub, mW]).transpose()
        return allocate


    def get_feedback(self, allocation, num_send_packet, power):
        self.rate = self.compute_instant_rate(allocation, power)
        l_max = np.floor(np.multiply(self.rate, self.T/self.D))

        num_received_packet = np.minimum(num_send_packet, l_max)
        
        self.packet_loss_rate = self.compute_packet_loss_rate(num_received_packet, num_send_packet)

        return num_received_packet
    
    def compute_h_sub(self, device_position, h_tilde):
        def path_loss_sub(distance):
            return 38.5 + 30*(np.log10(distance))
        
        h = np.abs(h_tilde* pow(10, -path_loss_sub(distance=np.linalg.norm(device_position))/20.0))**2

        return h
    
    def compute_h_mW(self, device_position, device_index, h_tilde, eta=5*np.pi/180, beta=0, epsilon=0.1):
        def path_loss_mW_los(distance):
            X = self.LOS_PATH_LOSS[self.current_step]
            return 61.4 + 20*(np.log10(distance))+X


        def path_loss_mW_nlos(distance):
            X = self.NLOS_PATH_LOSS[self.current_step]
            return 72 + 29.2*(np.log10(distance))+X
        
        # device blocked by obstacle
        if (device_index == 1 or device_index == 5):
            path_loss = path_loss_mW_nlos(distance=np.linalg.norm(device_position))
            h = G(eta, beta, epsilon)*pow(10, -path_loss/10)*epsilon # G_Rx^k=epsilon
        # device not blocked
        else:
            path_loss = path_loss_mW_los(distance=np.linalg.norm(device_position))
            h = G(eta, beta, epsilon)**2*pow(10, -path_loss/10) # G_Rx^k = G_b

        return h
    
    def compute_instant_rate(self, allocation, power):
        rate = np.zeros(shape=(self.num_devices, 2))
        
        for k in range(self.num_devices):
            sub_channel_index = allocation[k, 0]
            mW_beam_index = allocation[k, 1]
            if (sub_channel_index != -1):
                h_sub_k = self.compute_h_sub(
                    device_position=self.device_positions[k], 
                    h_tilde=self.h_tilde[self.current_step, 0, k, sub_channel_index]
                )

                p = power[k,0]*self.P_sum
                rate[k,0] = compute_rate_sub(h_sub_k, device_index=k,power=p)
            if (mW_beam_index != -1):
                h_mW_k = self.compute_h_mW(
                    device_position=self.device_positions[k], device_index=k, 
                    h_tilde=self.h_tilde[self.current_step, 1, k, mW_beam_index])
                p = power[k,1]*self.P_sum
                rate[k,1] = compute_rate_mW(h_mW_k, device_index=k,power=p)

        return rate
    
    def compute_average_rate(self):
        average_rate = 1/self.current_step*(self.rate + self.average_rate*(self.current_step-1))

        return average_rate

    def compute_packet_loss_rate(self, num_received_packet, num_send_packet):
        packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        for k in range(self.num_devices):
            if num_send_packet[k,0] > 0:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1) + (1 - num_received_packet[k,0]/num_send_packet[k,0]))
            else:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1))
            
            if num_send_packet[k,1] > 0:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1) + (1 - num_received_packet[k,1]/num_send_packet[k,1]))
            else:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1))

        return packet_loss_rate
    
    def estimate_average_channel_power(self, num_received_packet, power):
        estimate_instant_rate = num_received_packet*self.D/self.T
        average_channel_power = np.empty(shape=(self.num_devices, 2))

        for k in range(self.num_devices):
            if num_received_packet[k, 0] > 0:
                h = (2**(estimate_instant_rate[k,0]/W_SUB)-1)/power[k,0]*W_SUB*SIGMA_SQR
                average_channel_power[k,0] = 1/self.current_step*((self.current_step-1)*average_channel_power[k,0] + h)
            else:
                average_channel_power[k,0] = (self.current_step-1)/self.current_step*average_channel_power[k,0]
            
            if num_received_packet[k, 1] > 0:
                h = (2**(estimate_instant_rate[k,1]/W_SUB)-1)/power[k,1]*W_SUB*SIGMA_SQR
                average_channel_power[k,1] = 1/self.current_step*((self.current_step-1)*average_channel_power[k,1] + h)
            else:
                average_channel_power[k,1] = (self.current_step-1)/self.current_step*average_channel_power[k,1]

        return average_channel_power
    
    def step(self, policy_network_output):
        state = None
        reward = None
        terminated = False
        truncated = False
        info = {}

        num_send_packet, power, allocation = self.get_action(policy_network_output)
        num_received_packet = self.get_feedback(allocation, num_send_packet, power)
        self.average_rate = self.compute_average_rate()
        self.average_channel_power = self.estimate_average_channel_power(num_received_packet, power)

        reward_qos, reward_power, reward = self.get_reward(num_send_packet, num_received_packet, power)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("Reward is NaN or Inf")

        state = self.get_state(num_received_packet, power)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError("State contains NaN or Inf values")
        observation = state.flatten()
        

        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True

        info['Overall/ Reward'] = reward
        info['Overall/ Reward QoS'] = reward_qos
        info['Overall/ Reward Power'] = reward_power
        info['Overall/ Sum Packet loss rate'] = self.packet_loss_rate.sum()/(self.num_devices*2)
        info['Overall/ Average rate/ Sub6GHz'] = self.average_rate[:,0].sum()/(self.num_devices)
        info['Overall/ Average rate/ mmWave'] = self.average_rate[:,1].sum()/(self.num_devices)
        info['Overall/ Power usage'] = power.sum()
        
        for k in range(self.num_devices):
            info[f'Device {k+1}/ Num. Sent packet/ Sub6GHz'] = num_send_packet[k,0]
            info[f'Device {k+1}/ Num. Sent packet/ mmWave'] = num_send_packet[k,1]

            info[f'Device {k+1}/ Num. Received packet/ Sub6GHz'] = num_received_packet[k,0]
            info[f'Device {k+1}/ Num. Received packet/ mmWave'] = num_received_packet[k,1]
            
            info[f'Device {k+1}/ Num. Droped packet/ Sub6GHz'] = num_send_packet[k,0] - num_received_packet[k,0]
            info[f'Device {k+1}/ Num. Droped packet/ mmWave'] = num_send_packet[k,1] - num_received_packet[k,1]

            info[f'Device {k+1}/ Power/ Sub6GHz'] = power[k,0]
            info[f'Device {k+1}/ Power/ mmWave'] = power[k,1]

            info[f'Device {k+1}/ Packet loss rate/ Sub6GHz'] = self.packet_loss_rate[k,0]
            info[f'Device {k+1}/ Packet loss rate/ mmWave'] = self.packet_loss_rate[k,1]
            info[f'Device {k+1}/ Average rate/ Sub6GHz'] = self.average_rate[k,0]
            info[f'Device {k+1}/ Average rate/ mmWave'] = self.average_rate[k,1]
        

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        state = None
        info = {}
        self.current_step = 0
        self.state = np.zeros(shape=(self.num_devices, 8))
        state = self.state
        observation = state.flatten()
        self.action = np.zeros(shape=(self.num_devices, 4))
        self.instance_reward = 0.0
        self.reward_qos = 0.0
        self.current_step = 1

        self.average_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.packet_loss_rate = self._init_packet_loss_rate.copy()

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        return observation, info
    
    def get_reward(self, num_send_packet, num_received_packet, power):
        def calculate_efficiency_index(power, estimated_ideal_power, max_power=1.0):
            return (estimated_ideal_power - power)/max_power
        
        def estimate_ideal_power(num_send_packet, average_channel_power, W):
            if average_channel_power==0:
                return self.P_sum
            
            ideal_power = (2**((num_send_packet*self.D)/(W*self.T)) - 1) * \
                        W*SIGMA_SQR/average_channel_power
            return ideal_power
        
        reward_qos = 0
        reward_power = 0
        fairness_value = []

        for k in range(self.num_devices):
            prev_power_sub, prev_power_mW = self.state[k, 6], self.state[k, 7]
            power_sub, power_mw = power[k, 0], power[k, 1]
            qos_satisfaction = self.state[k, 0], self.state[k, 1]
            packet_loss_rate_sub, packet_loss_rate_mW = self.packet_loss_rate[k,0], self.packet_loss_rate[k,1]
            
            reward_qos += (num_received_packet[k,0] + num_received_packet[k,1])/(num_send_packet[k,0] + num_send_packet[k,1]) - (1-qos_satisfaction[0]) - (1-qos_satisfaction[1])

            if num_send_packet[k,0] > 0:
                estimated_ideal_power = estimate_ideal_power(num_send_packet[k,0], self.average_channel_power[k,0], W_SUB)
                eff_score = calculate_efficiency_index(power_sub, estimated_ideal_power)
                fairness_value.append(eff_score)

            if num_send_packet[k,1] > 0:
                estimated_ideal_power = estimate_ideal_power(num_send_packet[k,1], self.average_channel_power[k,1], W_SUB)
                eff_score = calculate_efficiency_index(power_sub, estimated_ideal_power)
                fairness_value.append(eff_score)

        fairness_value = np.array(fairness_value)
        reward_power = np.tanh(fairness_value.mean()/fairness_value.std())*self.num_devices
        reward_qos = ((self.current_step-1)*self.reward_qos + reward_qos)/self.current_step

        self.reward_qos = reward_qos
        self.instance_reward = self.reward_coef['reward_qos']*reward_qos + self.reward_coef['reward_power']*reward_power
        
        return reward_qos, reward_power, self.instance_reward
    

from gymnasium.envs.registration import register

register(
    id='PowerAllocationEnv-v1',
    entry_point='WirelessEnvironmentIntefaceOnly',
)