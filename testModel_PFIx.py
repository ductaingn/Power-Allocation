import numpy as np
from tqdm import tqdm
import Environment as env
import IO
import DDPGModel_PFix
import Model
import parameters

def train():
    # Initialize state, action, reward
    num_state = parameters.NUM_STATE
    num_action = parameters.NUM_ACTION
    model = DDPGModel_PFix.Brain(num_state, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,8))
    action = model.act(state,_notrandom=False)
    reward, instance_reward = 0,0

    # Load environment
    h_tilde = IO.load('h_tilde')
    device_positions = IO.load('device_positions')

    # Initialize variables
    number_of_send_packet = np.ones((env.NUM_OF_DEVICE,2))
    allocation = DDPGModel_PFix.allocate(number_of_send_packet)
    packet_loss_rate = np.zeros((env.NUM_OF_DEVICE,2))

    adverage_r = DDPGModel_PFix.compute_rate(device_positions,h_tilde[0],allocation,action,1)
    # rate = DDPGModel_PFix.compute_rate(device_positions,h_tilde[0],allocation,action,1)

    # Variables containt model's information in every frame
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
    LAMBDA = 0.99

    # Variables for reward based epsilon decay (not in use)
    MIN_VALUE = 0
    REWARD_TARGET = 10
    STEP_TO_TAKE = REWARD_TARGET
    REWARD_INCREMENT = 1
    REWARD_THRESHHOLD = 0
    CHANGE = (EPSILON - MIN_VALUE)/STEP_TO_TAKE
    # model.load_weights('./DDPG_weights/')

    for episode in tqdm(range(parameters.NUM_EPISODE)):
        # if(episode > 50 and instance_reward > REWARD_THRESHHOLD):
        #     EPSILON = max(0.1,EPSILON - CHANGE)
        #     REWARD_THRESHHOLD = REWARD_THRESHHOLD + REWARD_INCREMENT
        if(episode>30):
            EPSILON = max(0.01,EPSILON*LAMBDA)
        epsilon_plot.append(EPSILON)

        p = np.random.uniform(0,1,parameters.EPISODE_LENGTH)
        for frame in range(1,parameters.EPISODE_LENGTH):
            if(p[frame]>=EPSILON):
                # Greedy
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True)
            else:
                # Random
                action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=False)
            model.critic_network([np.expand_dims(state.flatten(),axis=0),np.expand_dims(action[:20], axis=0)])

            # Perform action
            l_max_estimate = DDPGModel_PFix.estimate_l_max(adverage_r,state,packet_loss_rate)
            number_of_send_packet = DDPGModel_PFix.test_compute_number_send_packet(action,l_max_estimate)
            number_of_send_packet_plot.append(number_of_send_packet)
            allocation = DDPGModel_PFix.allocate(number_of_send_packet)
            action_plot.append(action)

            # Get feedback
            r = DDPGModel_PFix.compute_rate(device_positions, h_tilde[frame], allocation, action,frame)
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
            reward, instance_reward = DDPGModel_PFix.compute_reward(state,action,number_of_send_packet,number_of_received_packet,reward,packet_loss_rate,frame)
            reward_plot.append(instance_reward)

            next_state = DDPGModel_PFix.update_state(packet_loss_rate,number_of_received_packet,action,adverage_r)

            # Add state, action, reward, next state, done into replay buffer
            model.remember(state.flatten(), instance_reward, next_state.flatten(), 0)
            batch = model.buffer.get_batch()
            # Update weights
            c_l, a_l = model.learn(batch)
            critic_loss.append(c_l)
            actor_loss.append(a_l)

            state = next_state

    IO.save(critic_loss,'critic_loss')
    IO.save(actor_loss,'actor_loss')
    model.save_weights('./DDPG_weights/')
    IO.save(reward_plot,'reward')
    IO.save(number_of_send_packet_plot,'number_of_sent_packet')
    IO.save(number_of_received_packet_plot,'number_of_received_packet')
    IO.save(packet_loss_rate_plot,'packet_loss_rate')
    IO.save(rate_plot,'rate')
    IO.save(action_plot,'action')
    IO.save(epsilon_plot,'epsilon')

def test():
    # Test
    num_state = parameters.NUM_STATE
    num_action = parameters.NUM_ACTION
    model = DDPGModel_PFix.Brain(num_state, num_action, 1, 0)
    state = np.zeros((env.NUM_OF_DEVICE,8))
    action = model.act(state,_notrandom=False)
    reward, instance_reward = 0,0

    h_tilde = IO.load('h_tilde')
    device_positions = IO.load('device_positions')

    number_of_send_packet = np.ones((env.NUM_OF_DEVICE,2))
    allocation = DDPGModel_PFix.allocate(number_of_send_packet)
    packet_loss_rate = np.zeros((env.NUM_OF_DEVICE,2))

    adverage_r = DDPGModel_PFix.compute_rate(device_positions,h_tilde[0],allocation,action,1)

    reward_plot = []
    packet_loss_rate_plot = []
    number_of_received_packet_plot = []
    number_of_send_packet_plot = []
    rate_plot = []
    action_plot = []

    model.load_weights('./DDPG_weights/')
    for frame in tqdm(range(251,10000)):
        action = model.act(np.expand_dims(state.flatten(),axis=0),_notrandom=True)

        l_max_estimate = DDPGModel_PFix.estimate_l_max(adverage_r,state,packet_loss_rate)
        number_of_send_packet = DDPGModel_PFix.test_compute_number_send_packet(action,l_max_estimate)
        number_of_send_packet_plot.append(number_of_send_packet)
        allocation = DDPGModel_PFix.allocate(number_of_send_packet)
        action_plot.append(action)


        # Get feedback
        r = DDPGModel_PFix.compute_rate(device_positions, h_tilde[frame], allocation, action,frame)
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
        reward, instance_reward = DDPGModel_PFix.compute_reward(state,action,number_of_send_packet,number_of_received_packet,reward,packet_loss_rate,frame)
        reward_plot.append(instance_reward)

        next_state = DDPGModel_PFix.update_state(packet_loss_rate,number_of_received_packet,action,adverage_r)

        state = next_state

    # model.save_weights('./DDPG_weights/')
    IO.save(reward_plot,'reward')
    IO.save(number_of_send_packet_plot,'number_of_sent_packet')
    IO.save(number_of_received_packet_plot,'number_of_received_packet')
    IO.save(packet_loss_rate_plot,'packet_loss_rate')
    IO.save(rate_plot,'rate')
    IO.save(action_plot,'action')

if __name__=="__main__":
    train()
