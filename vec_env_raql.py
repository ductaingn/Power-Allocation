import gymnasium as gym
from gymnasium import Env
from torch.nn.functional import softmax
from typing import Optional, Literal, Any, Tuple, Union
import pickle
import numpy as np
import torch
import wandb
from environment.Environment import r_sub as compute_rate_sub, r_mW as compute_rate_mW, G, W_SUB, W_MW, SIGMA_SQR
import random
from collections import defaultdict
import numpy as np

ln2 = np.log(2)

class Table:
    def __init__(self, default_value: float = 0.0):
        self.table = defaultdict(lambda: defaultdict(lambda: default_value))
        self.default_value = default_value

    def get(self, state: Any, action: Tuple[int, ...]) -> float:
        return self.table[state][action]

    def update(self, state: Any, action: Tuple[int, ...], value: float):
        self.table[state][action] = value

    def all_state_actions(self):
        for state, actions in self.table.items():
            for action, value in actions.items():
                yield (state, action, value)

    def save(self, filename):
        """
        Save the Q-table to a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def load(self, filename):
        """
        Load the Q-table from a file. This method is intended to be overridden by child classes.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def __add__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) + other.get(state, action))
        return result

    def __sub__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        result = Table(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) - other.get(state, action))
        return result

    def __mul__(self, scalar: Union[int, float]) -> "Table":
        result = Table(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result
    
    def __truediv__(self, scalar: Union[int, float]) -> "Table":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "Table":
        return self.__mul__(scalar)

    def __iadd__(self, other: "Table") -> "Table":
        if not isinstance(other, Table):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self
    
    def __pow__(self, exponent: float) -> "Table":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        
        result = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value ** exponent)
        return result

    def copy(self) -> "Table":
        new_q = Table(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        return new_q

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Table):
            return False
        return dict(self.table) == dict(other.table)

    def __repr__(self) -> str:
        entries = list(self.all_state_actions())
        preview = entries[:5]
        repr_str = "\n".join(f"{s} | {a} → {q:.2f}" for s, a, q in preview)
        if len(entries) > 5:
            repr_str += f"\n... and {len(entries)-5} more entries"
        return repr_str or "Table(empty)"

class QTable(Table):
    def __init__(self, default_value = 0):
        super().__init__(default_value)
        self.best_action_cache = {}  # Cache: state -> best_action

    def update(self, state, action, value):
        super().update(state, action, value)

        # Update best action cache
        current_best = self.best_action_cache.get(state)
        if current_best is None or value > self.get(state, current_best):
            self.best_action_cache[state] = action

    def best_action(self, state: Any):
        return self.best_action_cache.get(state, None)

    def max_q_value(self, state: Any) -> float:
        best = self.best_action(state)
        if best is None:
            return self.default_value
        return self.get(state, best)
    
    def __add__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) + other.get(state, action))
        return result

    def __sub__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        result = QTable(default_value=self.default_value)
        keys = set()
        for s in self.table:
            for a in self.table[s]:
                keys.add((s, a))
        for s in other.table:
            for a in other.table[s]:
                keys.add((s, a))
        for state, action in keys:
            result.update(state, action, self.get(state, action) - other.get(state, action))
        return result

    def __mul__(self, scalar: Union[int, float]) -> "QTable":
        result = QTable(default_value=self.default_value * scalar)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value * scalar)
        return result
    
    def __truediv__(self, scalar: Union[int, float]) -> "QTable":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Q-table by zero.")
        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value / scalar)
        return result

    def __rmul__(self, scalar: Union[int, float]) -> "QTable":
        return self.__mul__(scalar)

    def __iadd__(self, other: "QTable") -> "QTable":
        if not isinstance(other, QTable):
            return NotImplemented
        for state, action, value in other.all_state_actions():
            new_value = self.get(state, action) + value
            self.update(state, action, new_value)
        return self
    
    def __pow__(self, exponent: float) -> "QTable":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        
        result = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            result.update(state, action, value ** exponent)
        return result
    
    def copy(self):
        new_q = QTable(default_value=self.default_value)
        for state, action, value in self.all_state_actions():
            new_q.update(state, action, value)
        new_q.best_action_cache = self.best_action_cache.copy()
        return new_q
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'best_action_cache': self.best_action_cache,
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])
        self.best_action_cache = data['best_action_cache']


class VTable(Table):
    def __init__(self, default_value = 0):
        super().__init__(default_value)
    
    def update(self, state, action):
        if not (state in self.table):
            self.table[state][action] = 1 
        else:
            self.table[state][action] += 1
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])

class AlphaTable(Table):
    def __init__(self, default_value = 0):
        super().__init__(default_value)

    def update(self, state, action, value):
        return super().update(state, action, value)
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                'table': dict(self.table),
                'default_value': self.default_value,
            }, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.default_value = data['default_value']
        self.table = defaultdict(lambda: defaultdict(lambda: self.default_value), data['table'])


class WirelessEnvironmentRiskAverseQLearning(Env):
    def __init__(self, h_tilde_path: str, devices_positions_path: str, L_max: int, T: int, D: int, qos_threshold: float, P_sum, max_steps: int, reward_coef:dict, seed: Optional[int] = None, algorithm: Optional[Literal["RAQL"]] = "RAQL"):
        super(WirelessEnvironmentRiskAverseQLearning, self).__init__()
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
        self.epsilon = 0.5
        self.gamma = 0.9
        self.lambda_p = 0.5
        self.beta = -0.5
        self.lambda_ = 0.995
        self.x0 = -1 

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
                np.zeros(shape=(self.num_devices), dtype=int), # Quality of Service Satisfaction of each device on Sub6GHz
                np.zeros(shape=(self.num_devices), dtype=int), # Quality of Service Satisfaction of each device on mmWave,
                np.zeros(shape=(self.num_devices), dtype=int), # Number of received packets of each device on Sub6GHz of previous time step
                np.zeros(shape=(self.num_devices), dtype=int), # Number of received packets of each device on mmWave of previous time step
            ]).transpose().flatten(),
            high=np.array([
                np.ones(shape=(self.num_devices), dtype=int), # Quality of Service Satisfaction of each device on Sub6GHz
                np.ones(shape=(self.num_devices), dtype=int), # Quality of Service Satisfaction of each device on mmWave,
                np.full(shape=(self.num_devices), fill_value=self.L_max, dtype=int), # Number of received packets of each device on Sub6GHz of previous time step
                np.full(shape=(self.num_devices), fill_value=self.L_max, dtype=int), # Number of received packets of each device on mmWave of previous time step,
            ]).transpose().flatten(),
            dtype=int
        )

        self.action_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices), dtype=int), # Number of packets to send of each device on Sub6GHz,
            ]).flatten(),
            high=np.array([
                np.full(shape=(self.num_devices), fill_value=2, dtype=int), # Number of packets to send of each device on Sub6GHz,
            ]).flatten(),
            dtype=int
        )

        self.state = np.zeros(shape=(self.num_devices, 4), dtype=int)
        self.action = np.zeros(shape=(self.num_devices, 1), dtype=int)
        self.instance_reward = 0.0
        self.reward_qos = 0.0
        self.num_q_table = 4
        self.Q_tables = [QTable() for i in range(self.num_q_table)]
        self.V_tables = [VTable() for i in range(self.num_q_table)]
        self.Alpha_tables = [AlphaTable() for i in range(self.num_q_table)]

        self._init_num_send_packet = np.ones(shape=(self.num_devices, 2))
        self._init_power = np.full(shape=(self.num_devices, 2), fill_value=self.P_sum/(self.num_devices*2))
        self._init_allocation = self.allocate(self._init_num_send_packet)
        self.channel_power_gain = np.zeros(shape=(self.num_devices, 2))
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
                power=self.P_sum
            ))
            maximum_rate[1] = max(maximum_rate[1],compute_rate_mW(
                h=1.0,
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

    def get_state(self, num_received_packet):
        state = np.zeros(shape=(self.num_devices, 4))
        # QoS satisfaction
        state[:, 0] = (self.packet_loss_rate[:, 0] <= self.qos_threshold).astype(float)
        state[:, 1] = (self.packet_loss_rate[:, 1] <= self.qos_threshold).astype(float)
        state[:, 2] = num_received_packet[:, 0].copy()
        state[:, 3] = num_received_packet[:, 1].copy()
        
        self.state = state

        return state
    
    def get_action(self, action):
        l_max_estimate = self.estimate_l_max()

        if self.algorithm == "RAQL":
            num_send_packet, power = self.compute_number_send_packet_and_power(l_max_estimate, action)
            allocation = self.allocate(num_send_packet)

        return num_send_packet, power, allocation

    def estimate_l_max(self):
        l = np.multiply(self.average_rate, self.T/self.D)
        l_max_estimate = np.floor(l)

        return l_max_estimate

    def compute_number_send_packet_and_power(self, l_max_estimate, action)->tuple[np.ndarray, np.ndarray]:
        number_of_send_packet = np.zeros(shape=(self.num_devices, 2))

        power = np.full(shape=(self.num_devices, 2), fill_value=1.0/(self.num_sub_channel + self.num_beam))

        for k in range(self.num_devices):
            if action[k] == 0:
                number_of_send_packet[k,0] = max(1, min(l_max_estimate[k,0], self.L_max))

            if action[k] == 1:
                number_of_send_packet[k,1] = max(1, min(l_max_estimate[k,1], self.L_max))

            if action[k] == 2:
                if l_max_estimate[k,1] < self.L_max:
                    number_of_send_packet[k,1] = max(1, l_max_estimate[k,1])
                    number_of_send_packet[k,0] = min(max(1, l_max_estimate[k,0]), self.L_max-number_of_send_packet[k,1])
                else:
                    number_of_send_packet[k,0] = 1
                    number_of_send_packet[k,1] = self.L_max-1

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
    
    def compute_h_mW(self, device_position, device_index, h_tilde):
        def path_loss_mW_los(distance):
            X = self.LOS_PATH_LOSS[self.current_step]
            return 61.4 + 20*(np.log10(distance))+X


        def path_loss_mW_nlos(distance):
            X = self.NLOS_PATH_LOSS[self.current_step]
            return 72 + 29.2*(np.log10(distance))+X
        
        # device blocked by obstacle
        if (device_index == 1 or device_index == 5):
            path_loss = path_loss_mW_nlos(distance=np.linalg.norm(device_position))
            epsilon = 0.005
            h = G()*pow(10, -path_loss/10)*epsilon # G_Rx^k=epsilon
        # device not blocked
        else:
            path_loss = path_loss_mW_los(distance=np.linalg.norm(device_position))
            h = G()**2*pow(10, -path_loss/10) # G_Rx^k = G_b

        return h
    
    def compute_instant_rate(self, allocation, power):
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
    
    def u(self, x):
        u = -np.exp(self.beta*x)
        return u
    
    def step(self, policy_network_output):
        state = None
        reward = None
        terminated = False
        truncated = False
        info = {}

        state = tuple(self.state.flatten().tolist())
        H = np.random.randint(0, self.num_q_table)
        Q_random = self.Q_tables[H]
        self.epsilon = self.epsilon * self.lambda_

        p = np.random.rand()
        if p<self.epsilon:
            action = self.action_space.sample()
            action = tuple(action.flatten().tolist())

        else:
            average_q_table = sum(self.Q_tables, start=QTable(default_value=0))/self.num_q_table
            risk_averse_Q:QTable = Q_random - self.lambda_p/(self.num_q_table-1)*(
                sum(
                    ((self.Q_tables[i] - average_q_table) ** 2 for i in range(self.num_q_table)),
                    start=QTable(default_value=0)
                )
            )
            action = risk_averse_Q.best_action(state=state)
            if action is None:
                # If no action is found, sample a random action
                action = self.action_space.sample()
                action = tuple(action.flatten().tolist())
            
            if not(action in state):
                risk_averse_Q.update(state, action, risk_averse_Q.default_value)
                
            if risk_averse_Q.table[state][action] < risk_averse_Q.default_value:
                action = self.action_space.sample()

        num_send_packet, power, allocation = self.get_action(action)
        num_received_packet = self.get_feedback(allocation, num_send_packet, power)
        self.average_rate = self.compute_average_rate()

        reward_qos, reward = self.get_reward(num_send_packet, num_received_packet)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("Reward is NaN or Inf")

        next_state = self.get_state(num_received_packet)
        next_state = tuple(next_state.flatten().tolist())

        J = np.random.poisson(1, self.num_q_table)
        for i in range(self.num_q_table):
            if J[i] == 1:
                self.V_tables[i].update(state, action)
                self.Alpha_tables[i].update(state, action, 1/(self.V_tables[i].get(state, action)))

                q_update_value = self.Q_tables[i].get(state, action) + \
                    self.Alpha_tables[i].get(state, action)* \
                    (
                        self.u(
                            reward + \
                            self.gamma*self.Q_tables[i].max_q_value(next_state)-self.Q_tables[i].get(state, action)
                        ) -\
                        self.x0
                    )

                self.Q_tables[i].update(state, action, q_update_value)

        state = self.get_state(num_received_packet)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError("State contains NaN or Inf values")
        observation = state.flatten()

        info['Overall/ Reward'] = reward
        info['Overall/ Reward QoS'] = reward_qos
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

            info[f'Device {k+1}/ Packet loss rate/ Global'] = self.packet_loss_rate[k].sum()/2
            info[f'Device {k+1}/ Packet loss rate/ Sub6GHz'] = self.packet_loss_rate[k,0]
            info[f'Device {k+1}/ Packet loss rate/ mmWave'] = self.packet_loss_rate[k,1]
            info[f'Device {k+1}/ Average rate/ Sub6GHz'] = self.average_rate[k,0]
            info[f'Device {k+1}/ Average rate/ mmWave'] = self.average_rate[k,1]

            info[f'Device {k+1}/ Estimated ideal power/ Sub6GHz'] = self.estimated_ideal_power[k,0]/self.P_sum
            info[f'Device {k+1}/ Estimated ideal power/ mmWave'] = self.estimated_ideal_power[k,1]/self.P_sum
        
        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        state = None
        info = {}
        self.current_step = 0
        self.state = np.zeros(shape=(self.num_devices, 4))
        state = self.state
        observation = state.flatten()
        self.action = np.zeros(shape=(self.num_devices, 2))
        self.instance_reward = 0.0
        self.reward_qos = 0.0
        self.current_step = 1

        self.average_rate = self._init_rate.copy()
        self.instant_rate = self._init_rate.copy()
        self.packet_loss_rate = self._init_packet_loss_rate.copy()
        self.estimated_ideal_power = np.zeros(shape=(self.num_devices, 2))
        self.channel_power_gain = np.zeros(shape=(self.num_devices, 2))

        # LoS Path loss - mmWave
        self.LOS_PATH_LOSS = np.random.normal(0, 5.8, self.max_steps+1)
        # NLoS Path loss - mmWave
        self.NLOS_PATH_LOSS = np.random.normal(0, 8.7, self.max_steps+1) 

        return observation, info
    
    def get_reward(self, num_sent_packet, num_received_packet):        
        reward_qos = 0

        for k in range(self.num_devices):
            qos_satisfaction = self.state[k, 0], self.state[k, 1]
            
            reward_qos += (num_received_packet[k,0] + num_received_packet[k,1])/(num_sent_packet[k,0] + num_sent_packet[k,1]) - (1-qos_satisfaction[0]) - (1-qos_satisfaction[1])
        reward_qos = ((self.current_step-1)*self.reward_qos + reward_qos)/self.current_step

        self.reward_qos = reward_qos
        self.instance_reward = self.reward_coef['reward_qos']*reward_qos
        
        return reward_qos, self.instance_reward
    

from gymnasium.envs.registration import register

register(
    id='PowerAllocationEnv-v2',
    entry_point='WirelessEnvironmentRiskAverseQLearning',
)