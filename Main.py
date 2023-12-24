import Enviroment as env
import IO
import numpy as np
import matplotlib.pyplot as plt
import time

# Number of APs
NUM_OF_AP = 1
# Number of Devices
NUM_OF_DEVICE = 3
# Number of Sub-6GHz channels, Number of MmWave beams
N = M = 4
# Maximum Packet Loss Rate (PLR Requirement)
RHO_MAX = 0.1
# L_k
L_k = 6
# Number of Frame
T = 10000
# Risk control
LAMBDA_P = 0.5
# Ultility function paremeter
BETA = -0.5
# Learning parameters
GAMMA = 0.9
EPSILON = 0.5
# Decay factor
LAMBDA = 0.995
# Number of Q-tables
I = 2
X0 = -1
# Number of levels of quantitized Transmit Power 
A = 10


# CREAT STATE
# State is a NUM_OF_DEVICE*5 matrix
# in which state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, 
#                   ACK feedback at t-1 on sub, ACK feedback at t-1 on mW, 
#                   Power level at t-1 on sub, Power level at t-1 on mW]
def initialize_state():
    state = np.zeros(shape=(NUM_OF_DEVICE, 6),dtype=int)
    return state

def update_state(state, packet_loss_rate, feedback, power_level):
    for k in range(NUM_OF_DEVICE):
        for i in range(2):
            # QoS satisfaction
            if (packet_loss_rate[k, i] <= RHO_MAX):
                state[k, i] = 1
            elif (packet_loss_rate[k, i] > RHO_MAX):
                state[k, i] = 0
            # Number of successfully delivered packet on each interface
            state[k, i+2] = feedback[k, i]

        state[k,4] = power_level[0][k]
        state[k,5] = power_level[1][k]
    return state

# CREATE ACTION
# Action is an ndarray in which action[k] = [interface, power level on sub, power level on mW]
def initialize_action():
    # Initialize random action at the beginning
    # Choose interface 
    interface = np.random.randint(0, 3, NUM_OF_DEVICE)
    
    # Choose power level
    power_level = np.random.randint(0, A, (NUM_OF_DEVICE,2))

    action = np.ndarray(shape=(NUM_OF_DEVICE,3),dtype=int)
    for i in range(NUM_OF_DEVICE):
        action[i] = interface[i],power_level[i,0],power_level[i,1]
    return action


def chose_action(state, Q_table):
    # Epsilon-Greedy
    p = np.random.rand()
    action = initialize_action()
    if (p < EPSILON):
        return action
    else:
        max_Q = -np.Infinity
        for i in Q_table:
            state_i = np.asarray(i)[:, 0:6]
            if ((np.allclose(state_i, state, rtol=0, atol=0)) & (Q_table.get(i) > max_Q)):
                max_Q = Q_table.get(i)
                i = np.array(i,dtype=int)
                action = i[:, 6:9]
        return action

# Chose which subchannel/beam to allocate
# allocate[0] = [index of subchannel device 0 allocate, subchannel device 1 allocate, subchannel device 2 allocate]
# allocate[1] = [index of beam device 0 allocate, beam device 1 allocate, beam device 2 allocate]
def allocate(action):
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

    for k in range(NUM_OF_DEVICE):
        if (action[k,0] == 0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index)
        if (action[k,0] == 1):
            rand_index = np.random.randint(len(rand_mW))
            mW[k] = rand_mW[rand_index]
            rand_mW.pop(rand_index)
        if (action[k,0] == 2):
            rand_sub_index = np.random.randint(len(rand_sub))
            rand_mW_index = np.random.randint(len(rand_mW))

            sub[k] = rand_sub[rand_sub_index]
            mW[k] = rand_mW[rand_mW_index]

            rand_sub.pop(rand_sub_index)
            rand_mW.pop(rand_mW_index)

    allocate = [sub, mW]
    return allocate

# Return an matrix in which number_of_packet[k] = [number of transmit packets on sub, number of transmit packets on mW]
def compute_number_of_send_packet(action, l_sub_max, l_mW_max):
    number_of_packet = np.zeros(shape=(NUM_OF_DEVICE, 2))
    for k in range(NUM_OF_DEVICE):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if (action[k,0] == 0):
            # If l_sub_max too small, sent 1 packet and get bad reward later
            number_of_packet[k, 0] = max(1, min(l_sub_max_k, L_k))
            number_of_packet[k, 1] = 0

        if (action[k,0] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = max(1, min(l_mW_max_k, L_k))

        if (action[k,0] == 2):
            if (l_mW_max_k < L_k):
                number_of_packet[k, 1] = l_mW_max_k
                number_of_packet[k, 0] = min(l_sub_max_k, L_k - l_mW_max_k)
            if (l_mW_max_k >= L_k):
                number_of_packet[k, 0] = 1
                number_of_packet[k, 1] = L_k - 1
    return number_of_packet

# Choose power level will provide for each device over its beam/subchannel so it satisfies the power constraint base on previous rate
# Return a matrix in which power_level[k] = [power level on sub, power level on mW]
def compute_power_level(action,rate):
    power_level_sub = action[:,1]
    power_level_mW = action[:,2]
    rate_sub,rate_mW = rate
    rate_sub = list(rate_sub)
    rate_mW = list(rate_mW)
    best_rate_device_sub = rate_sub.index(max(rate_sub))
    best_rate_device_mW = rate_mW.index(max(rate_mW))

    while(not env.power_constraint_satisfaction(power_level_sub)):
        if(power_level_sub[best_rate_device_sub]>1):
            power_level_sub[best_rate_device_sub]-=1
        else:
            rate_sub[best_rate_device_sub] = -1
            best_rate_device_sub = rate_sub.index(max(rate_sub))

    while(not env.power_constraint_satisfaction(power_level_mW)):
        if(power_level_mW[best_rate_device_mW]>1):
            power_level_mW[best_rate_device_mW]-=1
        else:
            rate_mW[best_rate_device_mW] = -1
            best_rate_device_mW = rate_mW.index(max(rate_mW))

    return [power_level_sub,power_level_mW]

# Return an matrix in which feedback[k] = [number of received packets on sub, number of received packets on mW]
def receive_feedback(num_of_send_packet, l_sub_max, l_mW_max):
    feedback = np.zeros(shape=(NUM_OF_DEVICE, 2))

    for k in range(NUM_OF_DEVICE):
        l_sub_k = num_of_send_packet[k, 0]
        l_mW_k = num_of_send_packet[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])

    return feedback


def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_packet_num, sent_packet_num):
    packet_loss_rate = np.zeros(shape=(NUM_OF_DEVICE, 2))
    for k in range(NUM_OF_DEVICE):
        # Packet Successfull Rate of device k over Sub-6GHz Interface
        l_sub_k = sent_packet_num[k, 0]
        l_mW_k = sent_packet_num[k, 1]

        if (l_sub_k == 0):
            packet_loss_rate[k, 0] = env.packet_loss_rate(
                frame_num, old_packet_loss_rate[k, 0], 0, l_sub_k)
        if (l_sub_k > 0):
            packet_successfull_rate_sub = env.packet_successful_rate(
                received_packet_num[k, 0], sent_packet_num[k, 0])
            packet_loss_rate[k, 0] = env.packet_loss_rate(
                frame_num, old_packet_loss_rate[k, 0], packet_successfull_rate_sub, l_sub_k)

        # Packet Successfull Rate of device k over MmWave Interface
        if (l_mW_k == 0):
            packet_loss_rate[k, 1] = env.packet_loss_rate(
                frame_num, old_packet_loss_rate[k, 1], 0, l_mW_k)
        if (l_mW_k > 0):
            packet_successfull_rate_mW = env.packet_successful_rate(
                received_packet_num[k, 1], sent_packet_num[k, 1])
            packet_loss_rate[k, 1] = env.packet_loss_rate(
                frame_num, old_packet_loss_rate[k, 1], packet_successfull_rate_mW, l_mW_k)

    return packet_loss_rate


# CREATE REWARD
# Compute reward of one pair of (state, action)
def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward, frame_num):
    sum = 0
    for k in range(NUM_OF_DEVICE):
        state_k = state[k]
        sum = sum + (num_of_received_packet[k, 0] + num_of_received_packet[k, 1])/(
            num_of_send_packet[k, 0] + num_of_send_packet[k, 1]) - (1 - state_k[0]) - (1-state_k[1])
    sum = (((frame_num - 1)*old_reward) + sum)/frame_num
    return sum


# CREATE MODEL
# A Q-table is a dictionary with key=tuple(state.apend(action)), value = Q(state,action)
def initialize_Q_tables():
    Q_tables = []
    for i in range(I):
        Q = {}
        Q_tables.append(Q)
    return Q_tables


def add_2_Q_tables(Q1, Q2):
    for item in Q2:
        if (item in Q1):
            Q1[item] += Q2[item]
        else:
            Q1.update({item: Q2[item]})
    return Q1


def average_Q_table(Q_tables):
    res = {}
    for i in range(len(Q_tables)):
        res = add_2_Q_tables(res, Q_tables[i])
    for i in res:
        res[i] = res[i]/I
    return res


def compute_risk_adverse_Q(Q_tables, random_Q_index):
    Q_random = Q_tables[random_Q_index]
    Q_adverage = average_Q_table(Q_tables)
    sum_sqrt = {}
    for i in range(I):
        minus_Q_adverage = {}
        for j in Q_adverage:
            minus_Q_adverage.update({j: -Q_adverage[j]})

        sub = add_2_Q_tables(Q_tables[i], minus_Q_adverage)

        for j in sub:
            sub[j] *= sub[j]

    sum_sqrt = add_2_Q_tables(sum_sqrt, sub)

    for i in sum_sqrt:
        sum_sqrt[i] = -sum_sqrt[i]*LAMBDA_P/(I-1)

    res = add_2_Q_tables(Q_random, sum_sqrt)
    return res


def u(x):
    return -np.exp(BETA*x)


def update_Q_table(Q_table, alpha, old_reward,state,action,next_state,Q_max_table):
    state_action = np.insert(state, 6, action.transpose(), axis=1)
    state_action = tuple([tuple(row) for row in state_action])
    if(not state_action in Q_table):
        Q_table.update({state_action:0})

    # Find max(Q(s(t+1),a)
    max_Q = 0
    next_state = tuple([tuple(row) for row in next_state])
    if(next_state in Q_max_table):
        max_Q = Q_max_table[next_state]

    if (alpha.get(state_action) == None):
        alpha_state_action = 0
    else:
        alpha_state_action = alpha.get(state_action)

    Q_table[state_action] = Q_table[state_action] + alpha_state_action * (u(old_reward + GAMMA *
            max_Q - Q_table[state_action]) - X0)
    
    state = tuple([tuple(row) for row in np.array(state)])
    if((not (state in Q_max_table)) or Q_table[state_action] > Q_max_table[state]):
        Q_max_table[state] = Q_table[state_action]


    return Q_table


def initialize_V():
    return initialize_Q_tables()


def update_V(V, Q_table):
    for i in Q_table:
        if (i in V):
            V[i] += 1
        else:
            V.update({i: 1})
    return V


def initialize_alpha():
    return initialize_Q_tables()


def update_alpha(alpha, V):
    for i in V:
        if (i in alpha):
            alpha[i] = 1/V[i]
        else:
            alpha.update({i: 1/V[i]})
    return alpha

# Set up environment
# Complex channel coefficient
def generate_h_tilde(mu, sigma, num_of_frame):
    h_tilde = []
    h_tilde_sub = env.generate_h_tilde(
        mu, sigma, num_of_frame*NUM_OF_DEVICE*env.NUM_OF_SUB_CHANNEL)
    h_tilde_mW = env.generate_h_tilde(
        mu, sigma, num_of_frame*NUM_OF_DEVICE*env.NUM_OF_BEAM)
    for frame in range(num_of_frame):
        h_tilde_sub_t = np.empty(
            shape=(NUM_OF_DEVICE, env.NUM_OF_SUB_CHANNEL), dtype=complex)
        for k in range(NUM_OF_DEVICE):
            for n in range(env.NUM_OF_SUB_CHANNEL):
                h_tilde_sub_t[k, n] = h_tilde_sub[frame*NUM_OF_DEVICE *
                                                  env.NUM_OF_SUB_CHANNEL + k*env.NUM_OF_SUB_CHANNEL + n]

        h_tilde_mW_t = np.empty(
            shape=(NUM_OF_DEVICE, env.NUM_OF_BEAM), dtype=complex)
        for k in range(NUM_OF_DEVICE):
            for n in range(env.NUM_OF_BEAM):
                h_tilde_mW_t[k, n] = h_tilde_mW[frame*NUM_OF_DEVICE *
                                                env.NUM_OF_BEAM + k*env.NUM_OF_BEAM + n]
        h_tilde_t = [h_tilde_sub_t, h_tilde_mW_t]
        h_tilde.append(h_tilde_t)
    return h_tilde

# Achievable rate
def compute_rate(device_positions, h_tilde, allocation,power_level_list):
    r = []
    r_sub = np.zeros(NUM_OF_DEVICE)
    r_mW = np.zeros(NUM_OF_DEVICE)
    h_tilde_sub = h_tilde[0]
    h_tilde_mW = h_tilde[1]
    for k in range(NUM_OF_DEVICE):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]
        if (sub_channel_index != -1):
            h_sub_k = env.compute_h_sub(
                device_positions,device_index=k,h_tilde=h_tilde_sub[k, sub_channel_index])
            p = env.POWER_SET[power_level_list[0][k]]
            r_sub[k] = env.r_sub(h_sub_k, device_index=k,power=p)
        if (mW_beam_index != -1):
            h_mW_k = env.compute_h_mW(device_positions, device_index=k,
                                      eta=5*np.pi/180, beta=0, h_tilde=h_tilde_mW[k, mW_beam_index])
            p = env.POWER_SET[power_level_list[0][k]]
            r_mW[k] = env.r_mW(h_mW_k, device_index=k,power=p)

    r.append(r_sub)
    r.append(r_mW)
    return r


def compute_average_r(adverage_r, last_r, frame_num):
    for k in range(NUM_OF_DEVICE):
        adverage_r[0][k] = frame_num *(last_r[0][k]+adverage_r[0][k]*(frame_num-1))
        adverage_r[1][k] = frame_num *(last_r[1][k]+adverage_r[1][k]*(frame_num-1))
    return adverage_r

# l_max = r*T/d
def compute_l_max(r):
    l = np.floor(np.multiply(r, env.T/env.D))
    return l


# TRAINING
# Read from old Q-tables

# Train with new data
device_positions = env.initialize_devices_pos()
# env.plot_position(ap_pos=env.AP_POSITION, device_pos=device_positions)
state = initialize_state()
action = initialize_action()
reward = 0
allocation = allocate(action)
power_level = compute_power_level(action,rate=[np.zeros(NUM_OF_DEVICE),np.zeros(NUM_OF_DEVICE)])
Q_tables = initialize_Q_tables()
Q_max_table = initialize_Q_tables()
V = initialize_V()
alpha = initialize_alpha()
packet_loss_rate = np.zeros(shape=(NUM_OF_DEVICE, 2))

# Generate h_tilde for all frame
h_tilde = generate_h_tilde(0, 1, T)
h_tilde_t = h_tilde[0]
adverage_r = compute_rate(device_positions, h_tilde_t,
                        allocation=allocate(action),power_level_list=power_level)
r = compute_rate(device_positions, h_tilde_t, allocation,power_level_list=power_level)

state_plot=[]
action_plot=[]
reward_plot=[]
number_of_sent_packet_plot=[]
number_of_received_packet_plot=[]
power_level_plot = []

chose_action_time = 0
perform_action_time = 0
feedback_time = 0
compute_reward_time = 0
update_Q_time = 0
run_time = 0

chose_action_time_plot = []
perform_action_time_plot = []
feedback_time_plot = []
compute_reward_time_plot = []
update_Q_time_plot = []

for frame in range(1, T):
    frame_start_time = time.time()
    # Random Q-table
    H = np.random.randint(0, I)
    risk_adverse_Q = compute_risk_adverse_Q(Q_tables, H)

    # Update EPSILON
    EPSILON = EPSILON * LAMBDA

    # Set up environment
    h_tilde_t = h_tilde[frame]
    state_plot.append(state)

    # Select action
    chose_action_start_time = time.time()
    action = chose_action(state, risk_adverse_Q)
    allocation = allocate(action)
    action_plot.append(action)
    chose_action_time += time.time()-chose_action_start_time

    # Perform action
    perform_action_start_time = time.time()
    l_max_estimate = compute_l_max(adverage_r)
    l_sub_max_estimate = l_max_estimate[0]
    l_mW_max_estimate = l_max_estimate[1]
    number_of_send_packet = compute_number_of_send_packet(
        action, l_sub_max_estimate, l_mW_max_estimate)
    power_level = compute_power_level(action,rate=adverage_r)
    power_level_plot.append(power_level)
    number_of_sent_packet_plot.append(number_of_send_packet)
    perform_action_time += time.time()-perform_action_start_time
    

    # Get feedback
    feedback_start_time = time.time()
    r = compute_rate(device_positions, h_tilde_t, allocation,power_level)
    l_max = compute_l_max(r)
    l_sub_max = l_max[0]
    l_mW_max = l_max[1]

    number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
    packet_loss_rate = compute_packet_loss_rate(
        frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
    number_of_received_packet_plot.append(number_of_received_packet)
    adverage_r = compute_average_r(adverage_r, r, frame)
    feedback_time += time.time() - feedback_start_time

    # Compute reward
    compute_reward_start_time = time.time()
    reward = compute_reward(state,number_of_send_packet,number_of_received_packet,reward,frame)
    reward_plot.append(reward)
    next_state = update_state(state, packet_loss_rate,number_of_received_packet,power_level)
    compute_reward_time += time.time() - compute_reward_start_time

    # Generate mask J
    J = np.random.poisson(1, I)
    for i in range(I):
        if (J[i] == 1):
            update_Q_start_time = time.time()
            Q_table = update_Q_table(Q_tables[i],alpha[i],reward,state,action,next_state,Q_max_table[i])
            V[i] = update_V(V[i], Q_table)
            alpha[i] = update_alpha(alpha[i], V[i])
            update_Q_time += time.time()-update_Q_start_time

    state = next_state

    run_time += time.time() - frame_start_time
    print('frame: ',frame)



IO.save(number_of_received_packet_plot,'number_of_received_packet')
IO.save(number_of_sent_packet_plot,'number_of_sent_packet')
IO.save(reward_plot,'reward')
IO.save(action_plot,'action')
IO.save(state_plot,'state')
IO.save(h_tilde,'h_tilde')
IO.save(device_positions,'device_positions')
IO.save(Q_tables,'Q_tables')
IO.save(reward,'all_reward')
IO.save(chose_action_time,'chose_action_time')
IO.save(perform_action_time,'perform_action_time')
IO.save(feedback_time,'feedback_time')
IO.save(compute_reward_time,'compute_reward_time')
IO.save(update_Q_time,'update_Q_time')
IO.save(run_time,'run_time')
IO.save(power_level_plot,'power_level')