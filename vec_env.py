import gymnasium as gym
from gymnasium import Env
from typing import Optional
import pickle
import numpy as np
import torch
import wandb
from Environment import r_sub as compute_rate_sub, r_mW as compute_rate_mW, G

class WirelessEnvironment(Env):
    def __init__(self, h_tilde_path:str, devices_positions_path:str, L_max:int, T:int, D:int, qos_threshold:float, P_sum, max_steps:int):
        super(WirelessEnvironment, self).__init__()
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

        self.observation_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices)), # Packet Loss Rate of each device on Sub6GHz
                np.zeros(shape=(self.num_devices)), # Packet Loss Rate of each device on mmWave,
                np.zeros(shape=(self.num_devices)), # Number of received packets of each device on Sub6GHz of previous time step
                np.zeros(shape=(self.num_devices)), # Number of received packets of each device on mmWave of previous time step
                np.zeros(shape=(self.num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.zeros(shape=(self.num_devices)), # Average Rate of each device on mmWave of previous time step
                np.zeros(shape=(self.num_devices)), # Power of each device on Sub6GHz on previous time step
                np.zeros(shape=(self.num_devices)), # Power of each device on mmWave on previous time step
            ]).transpose().flatten(),
            high=np.array([
                np.ones(shape=(self.num_devices)), # Packet Loss Rate of each device on Sub6GHz
                np.ones(shape=(self.num_devices)), # Packet Loss Rate of each device on mmWave,
                np.full(shape=(self.num_devices), fill_value=self.L_max), # Number of received packets of each device on Sub6GHz of previous time step
                np.full(shape=(self.num_devices), fill_value=self.L_max), # Number of received packets of each device on mmWave of previous time step,
                np.ones(shape=(self.num_devices)), # Average Rate of each device on Sub6GHz of previous time step
                np.ones(shape=(self.num_devices)), # Average Rate of each device on mmWave of previous time step,
                np.ones(shape=(self.num_devices)), # Power of each device on Sub6GHz on previous time step
                np.ones(shape=(self.num_devices)), # Power of each device on mmWave on previous time step,
            ]).transpose().flatten()
        )

        self.action_space = gym.spaces.Box(
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

        self.state = np.zeros(shape=(self.num_devices, 8))
        self.action = np.zeros(shape=(self.num_devices, 4))
        self.instance_reward = 0.0
        self.cumulative_reward = 0.0
        self.current_step = 1
        self.max_steps = max_steps

        self.average_rate = np.zeros(shape=(self.num_devices, 2))
        self.instant_rate = np.zeros(shape=(self.num_devices, 2))
        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        wandb.init(project='PowerAllocation')    

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
        state[:, 0] = self.packet_loss_rate[:, 0].copy()
        state[:, 1] = self.packet_loss_rate[:, 1].copy()
        state[:, 2] = num_received_packet[:, 0].copy()
        state[:, 3] = num_received_packet[:, 1].copy()
        state[:, 4] = self.average_rate[:, 0]*1e-7
        state[:, 5] = self.average_rate[:, 1]*1e-10
        state[:, 6] = power[:, 0].copy()
        state[:, 7] = power[:, 1].copy()
        
        self.state = state

        return state
    
    def get_action(self, network_output):
        l_max_estimate = self.estimate_l_max()

        num_send_packet, power = self.compute_number_send_packet_and_power(network_output, l_max_estimate)

        allocation = self.allocate(num_send_packet)

        return num_send_packet, power, allocation
    
    def estimate_l_max(self):
        # To-do: Might try without estimate_l_max
        l = np.multiply(self.average_rate, self.T/self.D)
        packet_successful_rate = np.ones(shape=(self.num_devices,2)) - self.packet_loss_rate
        l_max_estimate = np.floor(l*packet_successful_rate)

        return l_max_estimate

    def compute_number_send_packet_and_power(self, network_output, l_max_estimate)->tuple[np.ndarray, np.ndarray]:
        power_start_index = 2*self.num_devices
        interface_score = network_output[:power_start_index].reshape(self.num_devices, 2)
        interface_score = torch.softmax(torch.tensor(interface_score), dim=1).numpy()

        number_of_send_packet = np.minimum(
            interface_score*self.L_max,
            l_max_estimate
        )

        power = network_output[power_start_index:]
        power = torch.softmax(torch.tensor(power), dim=-1).numpy()
        power = power.reshape(self.num_devices, 2)

        for k in range(self.num_devices):
            if number_of_send_packet[k,0] + number_of_send_packet[k,1] == 0: # Force to send at least one packet on more powerful channel
                if power[k,0] > power[k,1]:
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


    def allocate(self, num_send_packet):
        '''
        Allocate subchannel and beam to each device randomly
        num_send_packet: Number of packets to send on each device
        allocation: [subchannel, beam]
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
    
    def compute_h_mW(self, device_position, device_index, h_tilde, eta=5*np.pi/180, beta=0):
        def path_loss_mW_los(distance):
            X = self.LOS_PATH_LOSS[self.current_step]
            return 61.4 + 20*(np.log10(distance))+X


        def path_loss_mW_nlos(distance):
            X = self.NLOS_PATH_LOSS[self.current_step]
            return 72 + 29.2*(np.log10(distance))+X
        
        h = 0
        # device blocked by obstacle
        if (device_index == 1 or device_index == 5):
            path_loss = path_loss_mW_nlos(distance=np.linalg.norm(device_position))
            h = np.abs(G(eta, beta)*h_tilde* pow(10, -path_loss/20)*0.01)**2  # G_Rx^k=epsilon
        # device not blocked
        else:
            path_loss = path_loss_mW_los(distance=np.linalg.norm(device_position))
            h = np.abs((G(eta, beta)**2)*h_tilde * pow(10, -path_loss/20))**2  # G_Rx^k = G_b

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
            if num_received_packet[k,0] > 0:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1) + (1 - num_received_packet[k,0]/num_send_packet[k,0]))
            else:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1))
            
            if num_received_packet[k,1] > 0:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1) + (1 - num_received_packet[k,1]/num_send_packet[k,1]))
            else:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1))

        return packet_loss_rate

    def step(self, network_output):
        state = None
        reward = None
        terminated = False
        truncated = False
        info = {}

        num_send_packet, power, allocation = self.get_action(network_output)
        num_received_packet = self.get_feedback(allocation, num_send_packet, power)
        self.average_rate = self.compute_average_rate()

        reward = self.get_reward(num_send_packet, num_received_packet, power)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("Reward is NaN or Inf")

        state = self.get_state(num_received_packet, power)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError("State contains NaN or Inf values")
        observation = state.flatten()
        

        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True

        info['Number of send packet'] = num_send_packet
        info['Number of received packet'] = num_received_packet
        info['Power'] = power
        info['Packet loss rate'] = self.packet_loss_rate
        info['Average rate'] = self.average_rate

        wandb.log({
            "Overall/ Reward": reward,
            "Overall/ Sum Packet loss rate": self.packet_loss_rate.sum()/(self.num_devices*2),
            "Overall/ Power usage": power.sum(),
            "Overall / Average rate/ Sub6GHz": self.average_rate[:,0].sum()/(self.num_devices),
            "Overall / Average rate/ mmWave": self.average_rate[:,1].sum()/(self.num_devices)
        }, commit=False)

        # Log a bar chart of total dropped packets per device
        # Calculate dropped packets for each device and interface
        dropped_table = wandb.Table(columns=["Device", "Interface", "Dropped Packets"])
        for k in range(self.num_devices):
            dropped_sub6 = num_send_packet[k, 0] - num_received_packet[k, 0]
            dropped_mmW = num_send_packet[k, 1] - num_received_packet[k, 1]
            dropped_table.add_data(f"Device {k+1}", "Sub6GHz", dropped_sub6)
            dropped_table.add_data(f"Device {k+1}", "mmWave", dropped_mmW)

        # Create a bar chart from the table
        dropped_chart = wandb.plot.bar(dropped_table, "Device", "Dropped Packets", title="Dropped Packets per Device (Sub6GHz and mmWave)")
        wandb.log({"Dropped Packets Chart": dropped_chart})

        for k in range(self.num_devices):
            wandb.log({
                f'Device {k+1}/ Num. Sent packet/ Sub6GHz': num_send_packet[k,0],
                f'Device {k+1}/ Num. Sent packet/ mmWave': num_send_packet[k,1],

                f'Device {k+1}/ Num. Received packet/ Sub6GHz': num_received_packet[k,1],
                f'Device {k+1}/ Num. Received packet/ mmWave': num_received_packet[k,1],

                f'Device {k+1}/ Num. Droped packet/ Sub6GHz': num_send_packet[k,0] - num_received_packet[k,0],
                f'Device {k+1}/ Num. Droped packet/ mmWave': num_send_packet[k,1] - num_received_packet[k,1],

                f'Device {k+1}/ Power/ Sub6GHz': power[k,0],
                f'Device {k+1}/ Power/ mmWave': power[k,1],

                f'Device {k+1}/ Packet loss rate/ Sub6GHz': self.packet_loss_rate[k,0],
                f'Device {k+1}/ Packet loss rate/ mmWave': self.packet_loss_rate[k,1],

                f'Device {k+1}/ Average rate/ Sub6GHz': self.average_rate[k,0],
                f'Device {k+1}/ Average rate/ mmWave': self.average_rate[k,1],
            }, commit=True)

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
        self.cumulative_reward = 0.0
        self.current_step = 1

        self.average_rate = np.zeros(shape=(self.num_devices, 2))
        self.instant_rate = np.zeros(shape=(self.num_devices, 2))
        self.packet_loss_rate = np.zeros(shape=(self.num_devices, 2))

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        return observation, info
    
    def get_reward(self, num_send_packet, num_received_packet, power):
        reward_sum = 0
        reward_interface_risk = 0
        reward_power_risk = 0

        for k in range(self.num_devices):
            packet_loss_rate_sub, packet_loss_rate_mW = self.state[k, 0], self.state[k, 1]
            prev_power_sub, prev_power_mW = self.state[k, 6], self.state[k, 7]
            power_sub, power_mw = power[k, 0], power[k, 1]
            qos_satisfaction = [1,1]
            
            if packet_loss_rate_sub > self.qos_threshold:
                reward_interface_risk += self.num_devices*packet_loss_rate_sub
                qos_satisfaction[0] = 0
            if packet_loss_rate_mW > self.qos_threshold:
                reward_interface_risk += self.num_devices*packet_loss_rate_mW
                qos_satisfaction[1] = 0
            
            reward_sum += (num_received_packet[k,0] + num_received_packet[k,1])/(num_send_packet[k,0] + num_send_packet[k,1]) - (1-qos_satisfaction[0]) - (1-qos_satisfaction[1])
            reward_power_risk += self.sigmoid(
                -(power_sub - prev_power_sub)/(1 - prev_power_sub)*(1-packet_loss_rate_sub) + \
                -(power_mw - prev_power_mW)/(1 - prev_power_mW)*(1-packet_loss_rate_mW)
            )
            
            if np.isnan(reward_power_risk):
                
                raise ValueError(f"reward_power_risk resulted in NaN,(1 - prev_power_sub) = {(1 - prev_power_sub)}, 1 - prev_power_mW) = {(1 - prev_power_mW)}")

            if np.isinf(reward_power_risk):
                print(self.sigmoid(
                -(power_sub - prev_power_sub)/(1 - prev_power_sub)*(1-packet_loss_rate_sub) + \
                -(power_mw - prev_power_mW)/(1 - prev_power_mW)*(1-packet_loss_rate_mW)
                ))
                raise ValueError("reward_power_risk resulted in Inf")

        reward_sum = ((self.current_step-1)*self.cumulative_reward + reward_sum)/self.current_step

        self.cumulative_reward = reward_sum
        self.instance_reward = reward_sum - reward_interface_risk - reward_power_risk
        
        return self.cumulative_reward
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
from gymnasium.envs.registration import register

register(
    id='PowerAllocationEnv-v0',
    entry_point='WirelessEnvironment',
)