import gymnasium as gym
from gymnasium import Env
from torch.nn.functional import softmax
from typing import Optional, Literal
import pickle
import numpy as np
import torch
from environment.gym_env.Environment import r_sub as compute_rate_sub, r_mW as compute_rate_mW, G, gamma_sub, W_SUB, W_MW, SIGMA_SQR
import random
import attrs

ln2 = np.log(2)

@attrs.define
class WirelessEnvironmentBase(Env):
    h_tilde_path: str
    devices_positions_path: str
    num_devices: int
    L_max: int
    T: int 
    D: int
    qos_threshold: float
    P_sum:float
    max_steps: int
    reward_coef:dict
    seed: Optional[int] = None
    algorithm: Optional[Literal["Random", "RAQL", "SACPF", "SACPA"]] = attrs.field(default="SACPA")

    current_step:int = attrs.field(default=1)
    instance_reward:float = attrs.field(default=0.0)
    reward_qos:float = attrs.field(default=0.0)

    def __attrs_post_init__(self):
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps + 1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps + 1) 

        with open(self.h_tilde_path, 'rb') as f:
            self.h_tilde = np.array(pickle.load(f))

        with open(self.devices_positions_path, 'rb') as f:
            self.device_positions = np.array(pickle.load(f))

        self.num_sub_channel = self.h_tilde.shape[-1] # implicitly defined
        self.num_beam = self.h_tilde.shape[-1]

        self._init_num_send_packet:np.ndarray = attrs.field(default=np.ones(shape=(self.num_devices, 2)))

        self._init_num_send_packet = np.ones(shape=(self.num_devices, 2))
        self._init_power = np.full(shape=(self.num_devices, 2), fill_value=self.P_sum/(self.num_devices*2))
        self._init_allocation = self.allocate(self._init_num_send_packet)

        self.channel_power_gain = np.zeros(shape=(self.num_devices, 2))

        self._init_rate = self.compute_instant_rate(
            allocation=self._init_allocation,
            power=self._init_power
        )
        self._estimated_ideal_power = np.zeros(shape=(self.num_devices, 2))
        
        self.average_rate = self._init_rate
        self.previous_rate = self._init_rate.copy() # Data rate of previous time step
        self.instant_rate = self._init_rate.copy() # Data rate of current time step (acknowledge after get feedback)

        self.maximum_rate = np.array([
                compute_rate_sub(h=1.0, power=self.P_sum),
                compute_rate_mW(h=1.0, power=self.P_sum)
            ]) # For normalizing rate to [0,1]

        self.packet_loss_rate = np.zeros((self.num_devices, 2)) # Accumulated Packet loss rate of current time step of each device on each interface
        self.global_packet_loss_rate = np.zeros((self.num_devices)) # Accumulated Packet loss rate of current time step of each device over all interfaces
        self.sum_packet_loss_rate = 0 # Accumulated Packet loss rate of current time step of the whole system over all interfaces
        self._init_num_received_packet = self.get_feedback(self._init_allocation, self._init_num_send_packet, self._init_power)
        self._init_packet_loss_rate, self._init_global_packet_loss_rate, self._init_sum_packet_loss_rate = self.compute_packet_loss_rate(
            self._init_num_received_packet,
            self._init_num_send_packet,
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        if not hasattr(self, '_observation_space'):
            raise NotImplementedError("Subclasses must define 'observation_space' attribute.")
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value:gym.spaces.Space):
        self._observation_space = value

    @property
    def action_space(self) -> gym.spaces.Space:
        if not hasattr(self, '_action_space'):
            raise NotImplementedError("Subclasses must define 'action_space' attribute.")
        return self._action_space
        
    @action_space.setter
    def action_space(self, value:gym.spaces.Space):
        self._action_space = value

    @property
    def state(self) -> np.ndarray:
        if not hasattr(self, '_state'):
            raise NotImplementedError("Subclasses must define 'state' attribute.")
        return self._state
    
    @state.setter
    def state(self, value:np.ndarray):
        self._state = value

    @property
    def action(self):
        if not hasattr(self, '_action'):
            raise NotImplementedError("Subclasses must define 'action' attribute.")
        return self._action

    def get_state(self, **kwargs):
        raise NotImplementedError
    
    def get_action(self, value) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns number of sending packet, power, allocation(map devices with subchannel/beam)
        '''
        raise NotImplementedError
    
    def estimate_l_max(self):
        # To-do: Might try without estimate_l_max
        l = np.multiply(self.average_rate, self.T/self.D)
        packet_successful_rate = np.ones(shape=(self.num_devices,2)) - self.packet_loss_rate
        l_max_estimate = np.floor(l*packet_successful_rate)

        return l_max_estimate

    def compute_number_send_packet_and_power(self, **kwargs)->tuple[np.ndarray, np.ndarray]:
        '''
        Returns number of sending packet, power
        '''
        raise NotImplementedError

    def allocate(self, num_send_packet:np.ndarray) -> np.ndarray:
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

    def get_feedback(self, allocation:np.ndarray, num_send_packet:np.ndarray, power:np.ndarray) -> np.ndarray:
        '''
        Returns number of received packet at device side
        '''
        self.rate = self.compute_instant_rate(allocation, power)
        l_max = np.floor(np.multiply(self.rate, self.T/self.D))

        num_received_packet = np.minimum(num_send_packet, l_max)
        
        self.packet_loss_rate, self.global_packet_loss_rate, self.sum_packet_loss_rate = self.compute_packet_loss_rate(num_received_packet, num_send_packet)

        return num_received_packet
    
    def compute_h_sub(self, device_position, h_tilde):
        def path_loss_sub(distance):
            return 38.5 + 30*(np.log10(distance))
        
        h = np.abs(h_tilde* pow(10, -path_loss_sub(distance=np.linalg.norm(device_position))/20.0))**2

        return h
    
    def compute_h_mW(self, device_position, device_index, h_tilde):
        def path_loss_mW_los(distance):
            X = self.LOS_PATH_LOSS[self.current_step]
            return 61.4 + 20*(np.log10(distance))+X


        def path_loss_mW_nlos(distance):
            X = self.NLOS_PATH_LOSS[self.current_step]
            return 72 + 29.2*(np.log10(distance))+X
        
        # device blocked by obstacle
        if (device_index in [1,5,10,13]):
            path_loss = path_loss_mW_nlos(distance=np.linalg.norm(device_position))
            epsilon = 0.005 # side lobe beam gain
            h = G()*pow(10, -path_loss/10)*epsilon # G_Rx^k=epsilon
        # device not blocked
        else:
            path_loss = path_loss_mW_los(distance=np.linalg.norm(device_position))
            h = G()**2*pow(10, -path_loss/10) # G_Rx^k = G_b

        return h
    
    def compute_instant_rate(self, allocation:np.ndarray, power:np.ndarray):
        rate = np.zeros(shape=(self.num_devices, 2))
        
        for k in range(self.num_devices):
            sub_channel_index = allocation[k, 0]
            mW_beam_index = allocation[k, 1]
            if (sub_channel_index != -1):
                self.channel_power_gain[k, 0] = self.compute_h_sub(
                    device_position=self.device_positions[k], 
                    h_tilde=self.h_tilde[self.current_step, 0, k, sub_channel_index]
                )

                p = power[k,0]*self.P_sum
                rate[k,0] = compute_rate_sub(self.channel_power_gain[k, 0], power=p)
            if (mW_beam_index != -1):
                self.channel_power_gain[k, 1] = self.compute_h_mW(
                    device_position=self.device_positions[k], device_index=k, 
                    h_tilde=self.h_tilde[self.current_step, 1, k, mW_beam_index])
                p = power[k,1]*self.P_sum
                rate[k,1] = compute_rate_mW(self.channel_power_gain[k, 1], power=p)

        return rate
    
    def compute_average_rate(self) -> np.ndarray:
        average_rate = 1/self.current_step*(self.rate + self.average_rate*(self.current_step-1))

        return average_rate

    def compute_packet_loss_rate(self, num_received_packet:np.ndarray, num_send_packet:np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        '''
        Returns devices packet loss rate on each interfaces, devices packet loss rate on the whole, and system packet loss rate
        '''
        packet_loss_rate = np.zeros(shape=(self.num_devices, 2))
        global_packet_loss_rate = np.zeros(shape=(self.num_devices))
        for k in range(self.num_devices):
            if num_send_packet[k,0] > 0:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1) + (1 - num_received_packet[k,0]/num_send_packet[k,0]))
            else:
                packet_loss_rate[k,0] = 1/self.current_step*(self.packet_loss_rate[k,0]*(self.current_step-1))
            
            if num_send_packet[k,1] > 0:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1) + (1 - num_received_packet[k,1]/num_send_packet[k,1]))
            else:
                packet_loss_rate[k,1] = 1/self.current_step*(self.packet_loss_rate[k,1]*(self.current_step-1))

            global_packet_loss_rate[k] = 1/self.current_step*(self.global_packet_loss_rate[k]*(self.current_step-1) + (1 - (num_received_packet[k,0] + num_received_packet[k,1])/(num_send_packet[k,0] + num_send_packet[k,1])))

        sum_packet_loss_rate = 1/self.current_step*(self.sum_packet_loss_rate*(self.current_step-1) + (1-num_received_packet.sum()/num_send_packet.sum()))

        return packet_loss_rate, global_packet_loss_rate, sum_packet_loss_rate
    
    def estimate_average_channel_power(self, num_sent_packet, power, allocation):
        # for k in range(self.num_devices):
        #     if num_sent_packet[k, 0] > 0:
        #         sub_channel_index = allocation[k,0]
        #         # self.estimated_channel_power[k,0] = W_SUB*SIGMA_SQR*(2**(self.rate[k,0]/W_SUB))/(power[k,0]*self.P_sum)
        #         self.channel_power[k,0] = self.compute_h_sub(self.device_positions[k], self.h_tilde[self.current_step, 0, k, sub_channel_index])
            
        #     if num_sent_packet[k, 1] > 0:
        #         mW_beam_index = allocation[k,1]
        #         self.channel_power[k,1] = self.compute_h_mW(
        #             device_position=self.device_positions[k], device_index=k, 
        #             h_tilde=self.h_tilde[self.current_step, 1, k, mW_beam_index])

        return self.channel_power_gain
    
    def get_info(self, 
        reward:float, 
        reward_qos:float, 
        power:np.ndarray,
        num_sent_packet:np.ndarray,
        num_received_packet:np.ndarray,
        reward_power:float=None
    ) -> dict:
        info = {}

        info['Overall/ Reward'] = reward
        info['Overall/ Reward QoS'] = reward_qos
        info['Overall/ Reward Power'] = reward_power
        info['Overall/ Sum Packet loss rate'] = self.sum_packet_loss_rate
        info['Overall/ Average rate/ Sub6GHz'] = self.average_rate[:,0].sum()/(self.num_devices)
        info['Overall/ Average rate/ mmWave'] = self.average_rate[:,1].sum()/(self.num_devices)
        info['Overall/ Average rate/ Global'] = info['Overall/ Average rate/ Sub6GHz'] + info['Overall/ Average rate/ mmWave']
        info['Overall/ Power usage'] = power.sum()
        
        for k in range(self.num_devices):
            info[f'Device {k+1}/ Num. Sent packet/ Sub6GHz'] = num_sent_packet[k,0]
            info[f'Device {k+1}/ Num. Sent packet/ mmWave'] = num_sent_packet[k,1]

            info[f'Device {k+1}/ Num. Received packet/ Sub6GHz'] = num_received_packet[k,0]
            info[f'Device {k+1}/ Num. Received packet/ mmWave'] = num_received_packet[k,1]
            
            info[f'Device {k+1}/ Num. Droped packet/ Sub6GHz'] = num_sent_packet[k,0] - num_received_packet[k,0]
            info[f'Device {k+1}/ Num. Droped packet/ mmWave'] = num_sent_packet[k,1] - num_received_packet[k,1]

            info[f'Device {k+1}/ Power/ Sub6GHz'] = power[k,0]
            info[f'Device {k+1}/ Power/ mmWave'] = power[k,1]

            info[f'Device {k+1}/ Packet loss rate/ Global'] = self.global_packet_loss_rate[k]
            info[f'Device {k+1}/ Packet loss rate/ Sub6GHz'] = self.packet_loss_rate[k,0]
            info[f'Device {k+1}/ Packet loss rate/ mmWave'] = self.packet_loss_rate[k,1]
            info[f'Device {k+1}/ Average rate/ Sub6GHz'] = self.average_rate[k,0]
            info[f'Device {k+1}/ Average rate/ mmWave'] = self.average_rate[k,1]

            if hasattr(self, '_estimated_ideal_power'):
                info[f'Device {k+1}/ Estimated ideal power/ Sub6GHz'] = self.estimated_ideal_power[k,0]/self.P_sum
                info[f'Device {k+1}/ Estimated ideal power/ mmWave'] = self.estimated_ideal_power[k,1]/self.P_sum
        
        return info

    def step(self, action):
        raise NotImplementedError
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        info = {}
        observation = self.state.flatten()
        self.instance_reward = 0.0
        self.reward_qos = 0.0
        self.current_step = 1

        self.average_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.packet_loss_rate = self._init_packet_loss_rate.copy()
        self.global_packet_loss_rate = self._init_global_packet_loss_rate.copy()
        self.sum_packet_loss_rate = self._init_sum_packet_loss_rate
        self.channel_power_gain = np.zeros(shape=(self.num_devices, 2))

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        return observation, info
    
    def get_reward(self, **kwargs):
        raise NotImplementedError


from gymnasium.envs.registration import register

register(
    id='PowerAllocationEnv-v0',
    entry_point='WirelessEnvironment',
)