import Enviroment as env
import Model
import IO
import DDPGModel
import numpy as np
import tensorflow as tf
from tqdm import tqdm

if __name__=="__main__":
    num_state = 2*(env.NUM_OF_DEVICE)
    num_action = 2*(env.NUM_OF_DEVICE)
    NUM_EPISODE = 2
    model = DDPGModel.Brain(env.NUM_OF_DEVICE*2, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,2))
    action = model.act(state,_notrandom=False)
    reward = 0

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

    for episode in range(NUM_EPISODE):
        EPSILON = 1
        LAMBDA = 0.995

        for frame in tqdm(range(1,2500+1)):
            p = np.random.uniform(0,1)
            if(p<EPSILON):
                # Random
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True)
            else:
                # Random
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=False)

            EPSILON = EPSILON * LAMBDA

            # Perform action
            l_max_estimate = Model.compute_l_max(adverage_r)
            l_sub_max_estimate = l_max_estimate[0]
            l_mW_max_estimate = l_max_estimate[1]
            number_of_send_packet = DDPGModel.compute_number_of_send_packet(action, l_max_estimate)
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
            reward, instance_reward = Model.compute_reward(state,action,number_of_send_packet,number_of_received_packet,reward,frame)
            reward_plot.append(reward)

            next_state = DDPGModel.update_state(packet_loss_rate)

            model.remember(state.flatten(), reward, next_state.flatten(), 0)
            batch = model.buffer.get_batch()
            model.learn(batch)

            state = next_state

    IO.save(reward_plot,'reward')
    IO.save(number_of_send_packet_plot,'number_of_sent_packet')
    IO.save(number_of_received_packet_plot,'number_of_received_packet')
    IO.save(packet_loss_rate_plot,'packet_loss_rate')
    IO.save(rate_plot,'rate')
    IO.save(action_plot,'action')