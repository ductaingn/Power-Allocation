from environment.gym_env.Environment import W_SUB, W_MW, SIGMA_SQR
from .wireless_env_base import WirelessEnvironmentBase, ln2
from utils.q_learning import QTable, VTable, AlphaTable
import gymnasium as gym
from gymnasium.envs.registration import register
from torch.nn.functional import softmax
import numpy as np
from typing import Optional
import attrs

@attrs.define
class WirelessEnvironmentRAQL(WirelessEnvironmentBase):
    num_q_table:int = attrs.field(default=4)
    epsilon:float = attrs.field(default=0.5)
    gamma:float = attrs.field(default=0.9)
    lambda_p:float = attrs.field(default=0.5)
    beta:float = attrs.field(default=-0.5)
    lambda_:float = attrs.field(default=0.995)
    x0:float = attrs.field(default=-1) 

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._observation_space = gym.spaces.Box(
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

        self._action_space = gym.spaces.Box(
            low=np.array([
                np.zeros(shape=(self.num_devices), dtype=int), # Number of packets to send of each device on Sub6GHz,
            ]).flatten(),
            high=np.array([
                np.full(shape=(self.num_devices), fill_value=2, dtype=int), # Number of packets to send of each device on Sub6GHz,
            ]).flatten(),
            dtype=int
        )

        self._state = np.zeros(shape=(self.num_devices, 4), dtype=int)
        self._action = np.zeros(shape=(self.num_devices, 1), dtype=int)
        self.Q_tables = [QTable() for i in range(self.num_q_table)]
        self.V_tables = [VTable() for i in range(self.num_q_table)]
        self.Alpha_tables = [AlphaTable() for i in range(self.num_q_table)]

    def get_state(self, num_received_packet:np.ndarray) -> np.ndarray:
        _state = np.zeros(shape=(self.num_devices, 4))
        # QoS satisfaction
        _state[:, 0] = (self.packet_loss_rate[:, 0] <= self.qos_threshold).astype(float)
        _state[:, 1] = (self.packet_loss_rate[:, 1] <= self.qos_threshold).astype(float)
        _state[:, 2] = num_received_packet[:, 0].copy()
        _state[:, 3] = num_received_packet[:, 1].copy()
        
        return _state
    
    def get_action(self, action):
        l_max_estimate = self.estimate_l_max()

        if self.algorithm == "RAQL":
            num_send_packet, power = self.compute_number_send_packet_and_power(l_max_estimate, action)
            allocation = self.allocate(num_send_packet)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return num_send_packet, power, allocation

    def compute_number_send_packet_and_power(self, l_max_estimate:np.ndarray, action:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def u(self, x):
        u = -np.exp(self.beta*x)
        return u
    
    def step(self, policy_network_output):
        state = None
        reward = None
        terminated = False
        truncated = False

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

        info = self.get_info(
            reward=reward,
            reward_qos=reward_qos,
            power=power,
            num_sent_packet=num_send_packet,
            num_received_packet=num_received_packet
        )
        
        self.current_step += 1
        if self.current_step > self.max_steps:
            terminated = True

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._state = np.zeros(shape=(self.num_devices, 4))
        self._action = np.zeros(shape=(self.num_devices, 2))

        return super().reset(seed=seed)
    
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
    id='WirelessEnvironmentRAQL-v1',
    entry_point='WirelessEnvironmentRAQL',
)