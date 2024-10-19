"""
The main model declaration
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import AdamW

from common_definitions import (
    GAMMA, TAU,
    STD_DEV, BUFFER_SIZE, BATCH_SIZE,
    CRITIC_LR, ACTOR_LR, CONFIDENCE
)

from utilsPytorch import OUActionNoise, ReplayBuffer

import Environment as env

class MultiheadAttention(nn.Module):
    def __init__(self, dmodel, dk, dv, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dmodel = dmodel

        self.proj_q, self.bias_q = self._get_proj_bias(dk)
        self.proj_k, self.bias_k = self._get_proj_bias(dk)
        self.proj_v, self.bias_v = self._get_proj_bias(dv)
        
        self.output_proj = nn.Linear(dv * num_heads, dmodel, bias=False)

        self.register_buffer('scale', torch.tensor(dk, dtype=float).sqrt())
    
    def _get_proj_bias(self, hidsize):
        proj = nn.Parameter(torch.Tensor(self.num_heads, self.dmodel, hidsize))
        bias = nn.Parameter(torch.Tensor(1, self.num_heads, 1, hidsize))
        nn.init.xavier_uniform_(proj)
        nn.init.constant_(bias, 0.)
        return proj, bias

    def forward(self, q, k, v):
        # batch, seqlen, dmodel
        q = (q.unsqueeze(1) @ self.proj_q) + self.bias_q
        k = (k.unsqueeze(1) @ self.proj_k) + self.bias_k
        v = (v.unsqueeze(1) @ self.proj_v) + self.bias_v
        # batch, head, seqlen, dk|dv

        q, k, v = q.unsqueeze(3), k.unsqueeze(2), v.unsqueeze(2)
        # q: (batch, head, qlen, 1, dk)
        # k, v: (batch, head, 1, kvlen, dk|dv)
        logits = (q * k / self.scale).sum(-1, keepdim=True)
        # batch, head, qlen, kvlen, 1
        weighted_v = F.softmax(logits, -2) * v
        # batch, head, qlen, kvlen, dv
        heads = weighted_v.sum(-2)
        # batch, head, qlen, dv
        hid = torch.cat(heads.unbind(1), -1)
        # batch, qlen, dv * head
        output = self.output_proj(hid)
        # batch, qlen, dmodel
        return output

class ActorNetwork(nn.Module):
    '''
    Policy network
    '''
    def __init__(self, num_states=24, num_actions=4, *args, **kwargs) -> None:
        super(ActorNetwork, self).__init__(*args, **kwargs)
        self.num_states = num_states
        self.num_actions = num_actions

        self.reshape_size = num_states // env.NUM_OF_DEVICE

        # Layers
        self.embeding = nn.Linear(self.reshape_size, 256)
        self.attention = MultiheadAttention(dmodel=256, dk=256, dv=256, num_heads=3)
        self.compress = nn.Linear(256*env.NUM_OF_DEVICE, num_actions)
        self.power_fc = nn.Linear(num_actions, num_actions//2)
        self.interface_fc = nn.Linear(num_actions, num_actions//2)

        self.softmax = nn.Softmax(dim=-1)

        # Add LayerNorm layers
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # Reshape
        x = x.view(-1, env.NUM_OF_DEVICE, self.reshape_size)

        # Feature extraction
        x = self.embeding(x)
        x = torch.relu(x)

        # Multi-head attention 
        attention_out = self.attention(x, x, x)
        attention_out = self.layer_norm1(attention_out)

        # Add attention_out with embed
        x = x + attention_out
        x = self.layer_norm2(x)

        # Output layers
        x = x.flatten(start_dim=1)
        x = self.compress(x)
        x = torch.relu(x)

        power = self.power_fc(x)
        power = torch.relu(power)
        power = self.softmax(power)
        interface = self.interface_fc(x)
        interface = torch.relu(interface)
        interface = self.softmax(interface)
        
        return torch.cat([power, interface], dim=-1)
    
class CriticNetwork(nn.Module):
    '''
    Estimate Q-value network
    '''
    def __init__(self, num_states=24, num_actions=4, *args, **kwargs) -> None:
        super(CriticNetwork, self).__init__(*args, **kwargs)
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_reshape_dim = num_states // env.NUM_OF_DEVICE
        self.action_reshape_dim = num_actions // env.NUM_OF_DEVICE

        # Layers
        self.state_in = nn.Linear(self.state_reshape_dim, 128)
        self.action_in = nn.Linear(self.action_reshape_dim, 128)

        self.attention = MultiheadAttention(dmodel=256, dk=256, dv=256, num_heads=3)
        self.compress = nn.Linear(256*env.NUM_OF_DEVICE, num_actions)
        self.power_fc = nn.Linear(num_actions, num_actions//2)
        self.interface_fc = nn.Linear(num_actions, num_actions//2)

        self.value_out = nn.Linear(num_actions, 1)

        # Add LayerNorm layers
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)

    def forward(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        # State, action in
        state = state.view(-1, env.NUM_OF_DEVICE, self.state_reshape_dim)
        state = self.state_in(state)
        action = action.view(-1, env.NUM_OF_DEVICE, self.action_reshape_dim)
        action = self.action_in(action)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(x)

        # Multi-head attention 
        attention_out = self.attention(x, x, x)
        attention_out = self.layer_norm1(attention_out)

        # Add attention_out with embed
        x = x + attention_out
        x = self.layer_norm2(x)

        # Output layers
        x = x.flatten(start_dim=1)
        x = self.compress(x)
        x = torch.relu(x)

        power = self.power_fc(x)
        power = torch.relu(power)
        # power = self.softmax(power)
        interface = self.interface_fc(x)
        interface = torch.relu(interface)
        # interface = self.softmax(interface)

        out = torch.cat([power, interface], dim=-1)
        out = self.value_out(out)
        
        return out
    
class DDPGModel(nn.Module):
    def __init__(self, num_states=24, num_actions=4, gamma=GAMMA, tau=TAU, std_dev=STD_DEV, device='cpu', *args, **kwargs) -> None:
        super(DDPGModel, self).__init__(*args, **kwargs)
        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.actor_net = ActorNetwork(num_states, num_actions).to(self.device)
        self.actor_target = ActorNetwork(num_states, num_actions).to(self.device)
        self.critic_net = CriticNetwork(num_states, num_actions).to(self.device)
        self.critic_target = CriticNetwork(num_states, num_actions).to(self.device)

        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.actor_optimizer = AdamW(self.actor_net.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = AdamW(self.critic_net.parameters(), lr=CRITIC_LR)

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.gamma = gamma
        self.std_dev = std_dev
        self.tau = tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state:np.ndarray, _notrandom=False, noise=True):
        """
        Run action by the actor network

        Args:
            state: the current state
            _notrandom: whether greedy is used
            noise: whether noise is to be added to the result action (this improves exploration)

        Returns:
            the resulting action
        """
        if _notrandom:
            state = torch.Tensor(state).to(self.device)
            with torch.no_grad():
                self.cur_action = self.actor_net.forward(state)
            self.cur_action = self.cur_action.flatten().cpu().detach().numpy()
        else:
            def soft_max(x):
                return np.exp(x)/np.sum(np.exp(x),axis=0)
            self.cur_action = np.concatenate((
                soft_max(
                    np.random.uniform(0,1, self.num_actions//2)
                    + (self.noise() if noise else 0)
                    ),
                soft_max(
                    np.random.uniform(0,1, self.num_actions//2)
                    + (self.noise() if noise else 0)
                    )                  
            ),axis=0)

        # self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)
        return self.cur_action
    
    def remember(self, prev_state, reward, state, done):
        """
        Store states, reward, done value to the buffer
        """
        # record it in the buffer based on its reward
        self.buffer.append(prev_state, self.cur_action, reward, state, done)

    def update_params(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, sn:torch.Tensor, d:torch.Tensor):
        # with torch.no_grad():
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        target_q = reward + self.gamma*(1-d)*self.critic_target(sn, self.actor_target(sn))

        critic_loss = F.mse_loss(self.critic_net(state, action), target_q)
        
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (-self.critic_net(state, self.actor_net(state))).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        return critic_loss.item(), actor_loss.item()
    
    def learn(self, entry):
        s, a, r, sn, d = zip(*entry)
        critic_loss, actor_loss = self.update_params(
            torch.tensor(np.stack(s), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(a), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(r), dtype=torch.float32).to(self.device), 
            torch.tensor(np.stack(sn), dtype=torch.float32).to(self.device), 
            torch.tensor(np.stack(d), dtype=torch.float32).to(self.device)
        )

        return critic_loss, actor_loss
    
    def save_weights(self, file_path:str):
        """
        Save the DDPG model including the actor and critic networks, target networks,
        optimizers, and other parameters to a file.

        Args:
            file_path (str): Path where the model will be saved.
        """
        checkpoint = {
            'actor_net': self.actor_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'std_dev': self.std_dev,
            'buffer': self.buffer  # Saving buffer (optional, if needed)
        }
        
        torch.save(checkpoint, file_path + 'checkpoint.pt')
        print(f"Model saved to {file_path}")

    def load_weights(self, file_path: str):
        """
        Load the DDPG model from a saved checkpoint, including actor and critic networks,
        target networks, optimizers, and other parameters.

        Args:
            ddpg_model (DDPGModel): The DDPG model instance where the weights will be loaded.
            file_path (str): Path from which the model will be loaded.
        """
        checkpoint = torch.load(file_path + 'checkpoint.pt', map_location=self.device, weights_only=False)

        self.actor_net.load_state_dict(checkpoint['actor_net'])
        self.critic_net.load_state_dict(checkpoint['critic_net'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        # Restore hyperparameters
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.tau = checkpoint.get('tau', self.tau)
        self.std_dev = checkpoint.get('std_dev', self.std_dev)

        # Optionally, restore the buffer if saved (only if applicable)
        if 'buffer' in checkpoint:
            self.buffer = checkpoint['buffer']

        print(f"Model loaded from {file_path}")

def update_state(packet_loss_rate, num_received_packet, action, average_rate):
    r'''
    # state = {state_k} \forall k in K

    # state_k = {
                    packet_loss_rate_sub, packet_loss_rate_mW, 
                    number_received_packet_sub(t-1), number_received_packet_mW(t-1),
                    power_sub(t-1), power_mW(t-1)
                    avg_rate_sub(t-1), avg_rate_mW(t-1)
                }
    '''
    next_state = np.zeros((env.NUM_OF_DEVICE,8))
    for k in range(env.NUM_OF_DEVICE):
        for i in range(2):
            # if(packet_loss_rate[k,i]<=0.1):
            #     next_state[k,i] = 1
            # else:
            #     next_state[k,i] = 0
            next_state[k,i] = packet_loss_rate[k,i]
            next_state[k, i+2] = num_received_packet[k, i]
            next_state[k, i+4] = action[k*2+i]
        next_state[k, 6] = average_rate[0][k]*1e-7
        next_state[k, 7] = average_rate[1][k]*1e-10
    return next_state

def compute_number_of_send_packet(action, l_max_estimate, packet_loss_rate, L_k=6, confidence=CONFIDENCE):
    number_of_send_packet = np.zeros((env.NUM_OF_DEVICE,2))

    for k in range(env.NUM_OF_DEVICE):
        # Action is flattened
        confidence_sub = action[4*k+2]
        confidence_mw = action[4*k+3]
        sub_proportion = confidence_sub/(confidence_sub+confidence_mw) if(confidence_sub+confidence_mw>0) else 0.5
        
        if(sub_proportion>=confidence[1]):
            action[k*4] = confidence_sub + confidence_mw
            action[k*4+1] = 0
            number_of_send_packet[k,0] = max(1,min(int(l_max_estimate[0,k]*sub_proportion*packet_loss_rate[k,0]),L_k))
            number_of_send_packet[k,1] = 0

        elif(confidence[1]>sub_proportion>=confidence[0]):
            number_of_send_packet[k,0] = max(1,min(int(l_max_estimate[0,k]*sub_proportion*packet_loss_rate[k,0]),L_k))
            number_of_send_packet[k,1] = max(1,min(int(l_max_estimate[1,k]*(1-sub_proportion*packet_loss_rate[k,1])),L_k))

        else:
            number_of_send_packet[k,0] = 0
            number_of_send_packet[k,1] = max(1,min(int(l_max_estimate[1,k]*(1-sub_proportion)),L_k))
            action[k*4] = 0
            action[k*4+1] = confidence_sub+confidence_mw

    return number_of_send_packet


def test_compute_number_send_packet(action,l_max_estimate, L_k=6,confidence=CONFIDENCE):
    number_of_send_packet = np.zeros((env.NUM_OF_DEVICE,2))

    for k in range(env.NUM_OF_DEVICE):
        power_sub = action[2*k]
        power_mw = action[2*k + 1]
        conf_sub = action[2*env.NUM_OF_DEVICE + 2*k]
        conf_mw = action[2*env.NUM_OF_DEVICE + 2*k+1]
        pro_sub = conf_sub/(conf_mw+conf_sub)
       
        # Use sub6 only
        if(pro_sub>=confidence[1]):
            number_of_send_packet[k,0] = max(1,min([l_max_estimate[0,k],np.ceil(pro_sub*L_k)]))
            number_of_send_packet[k,1] = 0

        elif(confidence[1]>pro_sub>=confidence[0]):
            number_of_send_packet[k,0] = max(1,min([l_max_estimate[0,k],np.floor(pro_sub*L_k)]))
            number_of_send_packet[k,1] = max(1,min([l_max_estimate[1,k],np.ceil((1-pro_sub)*(L_k-number_of_send_packet[k,0]))]))

        else:
            number_of_send_packet[k,0] = 0
            number_of_send_packet[k,1] = max(1,min([l_max_estimate[1,k],np.ceil((1-pro_sub)*L_k)]))

        # Collect power
        if(number_of_send_packet[k,0]==0):
            action[2*k+1] += power_sub
            action[2*k] = 0
        if(number_of_send_packet[k,1]==0):
            action[2*k] += power_mw
            action[2*k+1] = 0

    return number_of_send_packet

def allocate(number_of_send_packet):
    sub = []  # Stores index of subchannel device will allocate
    mW = []  # Stores index of beam device will allocate
    for i in range(env.NUM_OF_DEVICE):
        sub.append(-1)
        mW.append(-1)

    rand_sub = []
    rand_mW = []
    for i in range(env.NUM_OF_SUB_CHANNEL):
        rand_sub.append(i)
    for i in range(env.NUM_OF_BEAM):
        rand_mW.append(i)

    for k in range(env.NUM_OF_DEVICE):
        if (number_of_send_packet[k,0]>0 and number_of_send_packet[k,1]==0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index)
        elif (number_of_send_packet[k,0]==0 and number_of_send_packet[k,1]>0):
            rand_index = np.random.randint(len(rand_mW))
            mW[k] = rand_mW[rand_index]
            rand_mW.pop(rand_index)
        else:
            rand_sub_index = np.random.randint(len(rand_sub))
            rand_mW_index = np.random.randint(len(rand_mW))

            sub[k] = rand_sub[rand_sub_index]
            mW[k] = rand_mW[rand_mW_index]

            rand_sub.pop(rand_sub_index)
            rand_mW.pop(rand_mW_index)

    allocate = np.array([sub, mW])
    return allocate

def compute_rate(device_positions, h_tilde, allocation, action,frame):
    r = []
    r_sub = np.zeros(env.NUM_OF_DEVICE)
    r_mW = np.zeros(env.NUM_OF_DEVICE)
    h_tilde_sub = h_tilde[0]
    h_tilde_mW = h_tilde[1]
    for k in range(env.NUM_OF_DEVICE):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]
        if (sub_channel_index != -1):
            h_sub_k = env.compute_h_sub(
                device_positions,device_index=k,h_tilde=h_tilde_sub[k, sub_channel_index])
            p = action[2*k]*env.P_SUM
            r_sub[k] = env.r_sub(h_sub_k, device_index=k,power=p)
        if (mW_beam_index != -1):
            h_mW_k = env.compute_h_mW(device_positions, device_index=k,
                                      eta=5*np.pi/180, beta=0, h_tilde=h_tilde_mW[k, mW_beam_index],frame=frame)
            p = action[2*k+1]*env.P_SUM
            r_mW[k] = env.r_mW(h_mW_k, device_index=k,power=p)

    r.append(r_sub)
    r.append(r_mW)
    return r

def sigmoid(x):
    # return 1/(1+np.exp(-200*env.NUM_OF_DEVICE*x))
    return 1/(1+np.exp(-env.NUM_OF_DEVICE*x))

def compute_reward(state, action, num_of_send_packet, num_of_received_packet, old_reward_value,packet_loss_rate, frame_num):
    sum = 0
    risk = 0
    interface_reward = 0
    power_risk = 0
    for k in range(env.NUM_OF_DEVICE):
        state_k = state[k]
        prev_pow_sub, prev_pow_mw = state_k[-4], state_k[-3]
        cur_pow_sub, cur_pow_mw = action[2*k],action[2*k+1]
        satisfaction = [0,0]
        if(state_k[0]<env.RHO_MAX):
            satisfaction[0] = 1
            if num_of_send_packet[k,0]>0:
                interface_reward += 0.5*(1-packet_loss_rate[k,0])
        else:
            risk += env.NUM_OF_DEVICE*packet_loss_rate[k,0]
        if(state_k[1]<env.RHO_MAX):
            satisfaction[1] = 1
            if num_of_send_packet[k,1]>0:
                interface_reward += 0.5*(1-packet_loss_rate[k,1])
        else:
            risk += env.NUM_OF_DEVICE*packet_loss_rate[k,1]
        sum = sum + (num_of_received_packet[k, 0] + num_of_received_packet[k, 1])/(
            num_of_send_packet[k, 0] + num_of_send_packet[k, 1]) - (1 - satisfaction[0]) - (1-satisfaction[1])
        
        power_risk = power_risk + sigmoid(
                        -(cur_pow_sub-prev_pow_sub)/(1-prev_pow_sub)*(1-packet_loss_rate[k,0]) + \
                        -(cur_pow_mw-prev_pow_mw)/(1-prev_pow_mw)*(1-packet_loss_rate[k,1])
                        # (cur_pow_sub-prev_pow_sub)/(1-prev_pow_sub)*packet_loss_rate[k,0] + \
                        # (cur_pow_mw-prev_pow_mw)/(1-prev_pow_mw)*packet_loss_rate[k,1]
        )
        
    sum = ((frame_num - 1)*old_reward_value + sum)/frame_num
    return [sum, sum-risk-power_risk + interface_reward]

# l_max = r*T/d
def estimate_l_max(r,state,packet_loss_rate):
    l = np.multiply(r, env.T/env.D)
    qos_violated = np.ones(shape=(env.NUM_OF_DEVICE,2)) - state[:,0:2]
    packet_successful_rate = np.ones(shape=packet_loss_rate.shape)-packet_loss_rate
    res = np.floor(l*packet_successful_rate.transpose()*qos_violated.transpose())
    return res