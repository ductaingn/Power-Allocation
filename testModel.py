import Enviroment as env
import Model
import IO
import DDPGModel
import numpy as np
import tensorflow as tf
from tqdm import tqdm

if __name__=="__main__":
    num_state = 6*(env.NUM_OF_DEVICE)
    num_action = 4*(env.NUM_OF_DEVICE)
    NUM_EPISODE = 250
    model = DDPGModel.Brain(num_state, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,6))
    action = model.act(state,_notrandom=False)
    reward, instance_reward = 0,0

    h_tilde = IO.load('h_tilde')
    # h_tilde = Model.generate_h_tilde(0,0.1,env.NUM_OF_FRAME+1)
    # IO.save(h_tilde,'h_tilde')
    device_positions = IO.load('device_positions')
    # device_positions = env.initialize_devices_pos()
    # IO.save(device_positions,'device_positions')

    number_of_send_packet = np.ones((env.NUM_OF_DEVICE,2))
    allocation = DDPGModel.allocate(number_of_send_packet)
    packet_loss_rate = np.zeros((env.NUM_OF_DEVICE,2))

    adverage_r = DDPGModel.compute_rate(device_positions,h_tilde[0],allocation,action,1)
    rate = DDPGModel.compute_rate(device_positions,h_tilde[0],allocation,action,1)

    reward_plot = []
    packet_loss_rate_plot = []
    number_of_received_packet_plot = []
    number_of_send_packet_plot = []
    rate_plot = []
    action_plot = []
    epsilon_plot = []

    critic_loss = []
    actor_loss = []

    EPSILON = 0.9
    LAMBDA = 0.99
    MIN_VALUE = 0
    REWARD_TARGET = 10
    STEP_TO_TAKE = REWARD_TARGET
    REWARD_INCREMENT = 1
    REWARD_THRESHHOLD = 0
    CHANGE = (EPSILON - MIN_VALUE)/STEP_TO_TAKE

    for episode in tqdm(range(NUM_EPISODE)):
        # if(episode > 50 and instance_reward > REWARD_THRESHHOLD):
        #     EPSILON = max(0.1,EPSILON - CHANGE)
        #     REWARD_THRESHHOLD = REWARD_THRESHHOLD + REWARD_INCREMENT
        if(episode>30):
            EPSILON = max(0.1,EPSILON*LAMBDA)
        epsilon_plot.append(EPSILON)

        p = np.random.uniform(0,1,21)
        for frame in range(1,21):
            if(p[frame]>=EPSILON):
                # Greedy
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True)
            else:
                # Random
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=False)

            # Perform action
            l_max_estimate = DDPGModel.estimate_l_max(adverage_r,state,packet_loss_rate)
            l_sub_max_estimate = l_max_estimate[0]
            l_mW_max_estimate = l_max_estimate[1]
            # number_of_send_packet = DDPGModel.compute_number_of_send_packet(action, l_max_estimate, packet_loss_rate)
            number_of_send_packet = DDPGModel.test_compute_number_send_packet(action)
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

            next_state = DDPGModel.update_state(packet_loss_rate,number_of_received_packet,action)

            model.remember(state.flatten(), instance_reward, next_state.flatten(), 0)
            batch = model.buffer.get_batch()
            c_l, a_l = model.learn(batch)
            critic_loss.append(c_l)
            actor_loss.append(a_l)

            state = next_state

    IO.save(reward_plot,'reward')
    IO.save(number_of_send_packet_plot,'number_of_sent_packet')
    IO.save(number_of_received_packet_plot,'number_of_received_packet')
    IO.save(packet_loss_rate_plot,'packet_loss_rate')
    IO.save(rate_plot,'rate')
    IO.save(action_plot,'action')
    IO.save(epsilon_plot,'epsilon')
    IO.save(critic_loss,'critic_loss')
    IO.save(actor_loss,'actor_loss')
    model.save_weights('./DDPG_weights')