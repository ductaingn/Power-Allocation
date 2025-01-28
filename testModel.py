import numpy as np
from tqdm import tqdm
import Environment as env
import IO
import DDPGModel
import Model
import parameters
from datetime import datetime
import yaml
import os

def train():
    # Load configurations
    with open('config.yaml','rt') as file:
        config = yaml.safe_load(file)
    prj_path = config['project_home']
    save_weight_every_n_iteration = config['train']['save_weight_every']
        
    # Initialize state, action, reward
    num_state = parameters.NUM_STATE
    num_action = parameters.NUM_ACTION
    model = DDPGModel.Brain(num_state, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,8))
    action = model.act(state,_notrandom=False)
    reward, instance_reward = 0,0

    # Load environment
    h_tilde = IO.load_h_tilde()
    device_positions = IO.load_positions()

    # Initialize variables
    number_of_send_packet = np.ones((env.NUM_OF_DEVICE,2))
    allocation = DDPGModel.allocate(number_of_send_packet)
    packet_loss_rate = np.zeros((env.NUM_OF_DEVICE,2))

    adverage_r = DDPGModel.compute_rate(device_positions,h_tilde[0],allocation,action,1)
    # rate = DDPGModel.compute_rate(device_positions,h_tilde[0],allocation,action,1)

    # Variables containt system information in every frame
    reward_plot = []
    packet_loss_rate_plot = []
    number_of_received_packet_plot = []
    number_of_send_packet_plot = []
    rate_plot = []
    action_plot = []
    epsilon_plot = []

    critic_loss = []
    actor_loss = []

    EPSILON = 1
    LAMBDA = 0.96

    # Variables for reward based epsilon decay (not in use)
    MIN_VALUE = 0
    REWARD_TARGET = 10
    STEP_TO_TAKE = REWARD_TARGET
    REWARD_INCREMENT = 1
    REWARD_THRESHHOLD = 0
    CHANGE = (EPSILON - MIN_VALUE)/STEP_TO_TAKE

    # Set up folders to save results and model weights
    iteration_count = 0     # Used for saving model weights
    formated_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')    
    new_weight_path = prj_path + 'DDPG_weights/' + formated_time + '/'
    new_results_path = prj_path + 'results/' + formated_time + '/'
    os.makedirs(new_weight_path)
    os.makedirs(new_results_path)

    for episode in tqdm(range(parameters.NUM_EPISODE)):
        # if(episode > 50 and instance_reward > REWARD_THRESHHOLD):
        #     EPSILON = max(0.1,EPSILON - CHANGE)
        #     REWARD_THRESHHOLD = REWARD_THRESHHOLD + REWARD_INCREMENT
        if(episode>30):
            EPSILON = max(0.01,EPSILON*LAMBDA)
        epsilon_plot.append(EPSILON)

        p = np.random.uniform(0,1,parameters.EPISODE_LENGTH)
        for frame in range(1,parameters.EPISODE_LENGTH):
            if iteration_count % save_weight_every_n_iteration == 0:
                model.save_weights(new_weight_path, iteration_count)
            iteration_count+=1

            if(p[frame]>=EPSILON):
                # Greedy
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True, noise=True)
            else:
                # Random
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=False, noise=True)

            # Perform action
            l_max_estimate = DDPGModel.estimate_l_max(adverage_r,state,packet_loss_rate)
            number_of_send_packet = DDPGModel.test_compute_number_send_packet(action,l_max_estimate)
            number_of_send_packet_plot.append(number_of_send_packet)
            allocation = DDPGModel.allocate(number_of_send_packet)
            action_plot.append(action)


            # Get feedback
            r = DDPGModel.compute_rate(device_positions, h_tilde[frame], allocation, action,frame)
            l_max = Model.compute_l_max(r)
            l_sub_max = l_max[0]
            l_mW_max = l_max[1]
            rate_plot.append(r)

            number_of_received_packet = Model.receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
            packet_loss_rate = Model.compute_packet_loss_rate(
                frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
            packet_loss_rate_plot.append(packet_loss_rate)
            number_of_received_packet_plot.append(number_of_received_packet)
            adverage_r = Model.compute_average_r(adverage_r, r, frame)

            # Compute reward
            reward, instance_reward = DDPGModel.compute_reward(state,action,number_of_send_packet,number_of_received_packet,reward,packet_loss_rate,frame)
            reward_plot.append(instance_reward)

            next_state = DDPGModel.update_state(packet_loss_rate,number_of_received_packet,action,adverage_r)

            # Add state, action, reward, next state, done into replay buffer
            model.remember(state.flatten(), instance_reward, next_state.flatten(), 0)
            batch = model.buffer.get_batch()
            # Update weights
            c_l, a_l = model.learn(batch)
            critic_loss.append(c_l)
            actor_loss.append(a_l)

            state = next_state

    model.save_weights(new_weight_path, iteration_count)
    IO.save(critic_loss, new_results_path + 'critic_loss.pickle')
    IO.save(actor_loss, new_results_path + 'actor_loss.pickle')
    IO.save(reward_plot, new_results_path + 'reward.pickle')
    IO.save(number_of_send_packet_plot, new_results_path + 'number_of_sent_packet.pickle')
    IO.save(number_of_received_packet_plot, new_results_path + 'number_of_received_packet.pickle')
    IO.save(packet_loss_rate_plot, new_results_path + 'packet_loss_rate.pickle')
    IO.save(rate_plot, new_results_path + 'rate.pickle')
    IO.save(action_plot, new_results_path + 'action.pickle')
    IO.save(epsilon_plot, new_results_path + 'epsilon.pickle')

def test():
    with open('config.yaml','rt') as file:
        config = yaml.safe_load(file)
    prj_path = config['project_home']
    weight_path = config['test']['DDPG_weight_path']
    # Test
    num_state = parameters.NUM_STATE
    num_action = parameters.NUM_ACTION
    model = DDPGModel.Brain(num_state, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,8))
    action = model.act(state,_notrandom=False)
    reward, instance_reward = 0,0

    h_tilde = IO.load_h_tilde('test')
    device_positions = IO.load_positions('test')

    number_of_send_packet = np.ones((env.NUM_OF_DEVICE,2))
    allocation = DDPGModel.allocate(number_of_send_packet)
    packet_loss_rate = np.zeros((env.NUM_OF_DEVICE,2))

    adverage_r = DDPGModel.compute_rate(device_positions,h_tilde[0],allocation,action,1)

    reward_plot = []
    packet_loss_rate_plot = []
    number_of_received_packet_plot = []
    number_of_send_packet_plot = []
    rate_plot = []
    action_plot = []

    formated_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')    
    new_results_path = prj_path + 'results/' + formated_time + '/'
    os.makedirs(new_results_path)

    # model.load_weights('./DDPG_weights/best_weights/')
    model.load_weights(prj_path + '/' + weight_path, config['test']['iteration_count'])
    for frame in tqdm(range(parameters.EPISODE_LENGTH+1,env.NUM_OF_FRAME)):
        action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True, noise=False)

        l_max_estimate = DDPGModel.estimate_l_max(adverage_r,state,packet_loss_rate)
        number_of_send_packet = DDPGModel.test_compute_number_send_packet(action,l_max_estimate)
        number_of_send_packet_plot.append(number_of_send_packet)
        allocation = DDPGModel.allocate(number_of_send_packet)
        action_plot.append(action)


        # Get feedback
        r = DDPGModel.compute_rate(device_positions, h_tilde[frame], allocation, action,frame)
        l_max = Model.compute_l_max(r)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        rate_plot.append(r)

        number_of_received_packet = Model.receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
        packet_loss_rate = Model.compute_packet_loss_rate(
            frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
        packet_loss_rate_plot.append(packet_loss_rate)
        number_of_received_packet_plot.append(number_of_received_packet)
        adverage_r = Model.compute_average_r(adverage_r, r, frame)

        # Compute reward
        reward, instance_reward = DDPGModel.compute_reward(state,action,number_of_send_packet,number_of_received_packet,reward,packet_loss_rate,frame)
        reward_plot.append(instance_reward)

        next_state = DDPGModel.update_state(packet_loss_rate,number_of_received_packet,action,adverage_r)

        state = next_state

    # model.save_weights('./DDPG_weights/')
    IO.save(reward_plot, new_results_path + 'reward.pickle')
    IO.save(number_of_send_packet_plot, new_results_path + 'number_of_sent_packet.pickle')
    IO.save(number_of_received_packet_plot, new_results_path + 'number_of_received_packet.pickle')
    IO.save(packet_loss_rate_plot, new_results_path + 'packet_loss_rate.pickle')
    IO.save(rate_plot, new_results_path + 'rate.pickle')
    IO.save(action_plot, new_results_path + 'action.pickle')

if __name__=="__main__":
    test()
