"""
The main model declaration
"""
import logging
import os

import numpy as np
import tensorflow as tf

from common_definitions import (
    KERNEL_INITIALIZER, GAMMA, RHO,
    STD_DEV, BUFFER_SIZE, BATCH_SIZE,
    CRITIC_LR, ACTOR_LR, CONFIDENCE
)
from buffer import ReplayBuffer
from utils import OUActionNoise

import Enviroment as env

def InitActorNetwork(num_states=24, num_actions=4, action_high=1):
    """
    Get Actor Network with the given parameters.

    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
        action_high: the top value of the action

    Returns:
        the Keras Model
    """
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_normal_initializer(stddev=0.0005)

    inputs = tf.keras.layers.Input(shape=(num_states,), dtype=tf.float32)
    out = tf.keras.layers.Dense(600, activation=tf.nn.relu,
                                kernel_initializer=KERNEL_INITIALIZER)(inputs)
    out = tf.keras.layers.Dense(300, activation=tf.nn.relu,
                                kernel_initializer=KERNEL_INITIALIZER)(out)
    out = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(
        out) * action_high
    
    outputs = tf.keras.layers.Softmax(axis=-1)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def InitCriticNetwork(num_states=24, num_actions=4, action_high=1):
    """
    Get Critic Network with the given parameters.

    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
        action_high: the top value of the action

    Returns:
        the Keras Model
    """
    last_init = tf.random_normal_initializer(stddev=0.00005)

    # State as input
    state_input = tf.keras.layers.Input(shape=(num_states), dtype=tf.float32)
    state_out = tf.keras.layers.Dense(600, activation=tf.nn.relu,
                                      kernel_initializer=KERNEL_INITIALIZER)(state_input)
    state_out = tf.keras.layers.BatchNormalization()(state_out)
    state_out = tf.keras.layers.Dense(300, activation=tf.nn.relu,
                                      kernel_initializer=KERNEL_INITIALIZER)(state_out)

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions), dtype=tf.float32)
    action_out = tf.keras.layers.Dense(300, activation=tf.nn.relu,
                                       kernel_initializer=KERNEL_INITIALIZER)(
        action_input / action_high)

    # Both are passed through seperate layer before concatenating
    added = tf.keras.layers.Add()([state_out, action_out])

    added = tf.keras.layers.BatchNormalization()(added)
    outs = tf.keras.layers.Dense(150, activation=tf.nn.relu,
                                 kernel_initializer=KERNEL_INITIALIZER)(added)
    outs = tf.keras.layers.BatchNormalization()(outs)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


class Brain:  # pylint: disable=too-many-instance-attributes
    """
    The Brain that contains all the models
    """

    def __init__(
        self, num_states, num_actions, action_high, action_low, gamma=GAMMA, rho=RHO,
        std_dev=STD_DEV
    ):  # pylint: disable=too-many-arguments
        # initialize everything
        self.actor_network = InitActorNetwork(num_states, num_actions, action_high)
        self.critic_network = InitCriticNetwork(num_states, num_actions, action_high)
        self.actor_target = InitActorNetwork(num_states, num_actions, action_high)
        self.critic_target = InitCriticNetwork(num_states, num_actions, action_high)

        # Making the weights equal initially
        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.gamma = tf.constant(gamma)
        self.rho = rho
        self.action_high = action_high
        self.action_low = action_low
        self.num_states = num_states
        self.num_actions = num_actions
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR, amsgrad=True)

        # temporary variable for side effects
        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_actions), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ])
        def update_weights(s, a, r, sn, d):
            """
            Function to update weights with optimizer
            """
            with tf.GradientTape() as tape:
                # define target
                y = r + self.gamma * (1 - d) * self.critic_target([sn, self.actor_target(sn)])
                # define the delta Q
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network([s, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables))
            return critic_loss, actor_loss

        self.update_weights = update_weights

    @staticmethod
    def _update_target(model_target, model_ref, rho=0):
        """
        Update target's weights with the given model reference

        Args:
            model_target: the target model to be changed
            model_ref: the reference model
            rho: the ratio of the new and old weights
        """
        model_target.set_weights(
            [
                rho * ref_weight + (1 - rho) * target_weight
                for (target_weight, ref_weight)
                in list(zip(model_target.get_weights(), model_ref.get_weights()))
            ]
        )

    def act(self, state, _notrandom=True, noise=True):
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
            self.cur_action = self.actor_network(state)[0].numpy()
        else:
            def soft_max(x):
                return np.exp(x)/np.sum(np.exp(x),axis=0)
            self.cur_action = soft_max(
                np.random.uniform(self.action_low, self.action_high, self.num_actions)
                + (self.noise() if noise else 0)
            )

        self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)
        return self.cur_action

    def remember(self, prev_state, reward, state, done):
        """
        Store states, reward, done value to the buffer
        """
        # record it in the buffer based on its reward
        self.buffer.append(prev_state, self.cur_action, reward, state, done)

    def learn(self, entry):
        """
        Run update for all networks (for training)
        """
        s, a, r, sn, d = zip(*entry)

        c_l, a_l = self.update_weights(tf.convert_to_tensor(s, dtype=tf.float32),
                                       tf.convert_to_tensor(a, dtype=tf.float32),
                                       tf.convert_to_tensor(r, dtype=tf.float32),
                                       tf.convert_to_tensor(sn, dtype=tf.float32),
                                       tf.convert_to_tensor(d, dtype=tf.float32))

        self._update_target(self.actor_target, self.actor_network, self.rho)
        self._update_target(self.critic_target, self.critic_network, self.rho)

        return c_l, a_l

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_network.save_weights(path + "cn.h5")
        self.critic_target.save_weights(path + "ct.h5")
        self.actor_target.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target.load_weights(path + "ct.h5")
            self.actor_target.load_weights(path + "at.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)

def update_state(packet_loss_rate):
    next_state = np.zeros((env.NUM_OF_DEVICE,2))
    for k in range(env.NUM_OF_DEVICE):
        for i in range(2):
            if(packet_loss_rate[k,i]<=0.1):
                next_state[k,i] = 1
            else:
                next_state[k,i] = 0
    return next_state

def compute_number_of_send_packet(action, l_max_estimate, L_k=6, confidence=CONFIDENCE):
    number_of_send_packet = np.zeros((env.NUM_OF_DEVICE,2))

    for k in range(env.NUM_OF_DEVICE):
        # Action is flattened
        p_sub_k = action[2*k]
        p_mW_k = action[2*k+1]
        p_sub_proportion = p_sub_k/(p_sub_k+p_mW_k)
        
        if(p_sub_proportion>=confidence[1]):
            action[k*2] = p_sub_k+p_mW_k
            action[k*2+1] = 0
            number_of_send_packet[k,0] = max(1,min(int(l_max_estimate[0,k]*p_sub_proportion),L_k))
            number_of_send_packet[k,1] = 0

        elif(confidence[1]>p_sub_proportion>=confidence[0]):
            number_of_send_packet[k,0] = max(1,min(int(l_max_estimate[0,k]*p_sub_proportion),L_k))
            number_of_send_packet[k,1] = max(1,min(int(l_max_estimate[1,k]*(1-p_sub_proportion)),L_k))

        else:
            number_of_send_packet[k,0] = 0
            number_of_send_packet[k,1] = max(1,min(int(l_max_estimate[1,k]*(1-p_sub_proportion)),L_k))
            action[k*2] = 0
            action[k*2+1] = p_sub_k+p_mW_k

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