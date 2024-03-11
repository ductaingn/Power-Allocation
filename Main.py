import Enviroment as env
import IO
import numpy as np
import matplotlib.pyplot as plt
import time

# Number of APs
NUM_OF_AP = 1
# Maximum Packet Loss Rate (PLR Requirement)
RHO_MAX = 0.1
# L_k
L_k = 6
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
I = 4
X0 = -1


# CREAT STATE
# State is a NUM_OF_DEVICE*5 matrix
# in which state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, 
#                   ACK feedback at t-1 on sub, ACK feedback at t-1 on mW, 
#                   Power level at t-1 on sub, Power level at t-1 on mW]
def initialize_state():
    state = np.zeros(shape=(env.NUM_OF_DEVICE, 6),dtype=int)
    return state

def update_state(packet_loss_rate, feedback, power_level):
    next_state = np.zeros(shape=(env.NUM_OF_DEVICE, 6),dtype=int)
    for k in range(env.NUM_OF_DEVICE):
        for i in range(2):
            # QoS satisfaction
            if (packet_loss_rate[k, i] <= RHO_MAX):
                next_state[k, i] = 1
            elif (packet_loss_rate[k, i] > RHO_MAX):
                next_state[k, i] = 0
            # Number of successfully delivered packet on each interface
            next_state[k, i+2] = feedback[k, i]

        next_state[k,4] = power_level[0][k]
        next_state[k,5] = power_level[1][k]
    return next_state

# CREATE ACTION
possible_actions = IO.load('possible_actions')
# Action is an ndarray in which action[k] = [interface, power level]
# Action is an array where action[k] is the action of device k
def initialize_action():
    # Initialize random action at the beginning
    action = possible_actions[np.random.randint(0,len(possible_actions))]
    return action

def choose_action(state, Q_table):
    # Epsilon-Greedy
    p = np.random.rand()
    action = initialize_action()
    if (p < EPSILON):
        return action
    else:
        max_Q = -np.Infinity
        state = tuple([tuple(row) for row in state])
        action = tuple([tuple(row) for row in action])
        random_action = []  # Containts action with Q_value = 0
        for a in Q_table[state]:
            if(Q_table[state][a]>=max_Q):
                max_Q = Q_table[state][a]
                action = a
                if(max_Q==0):
                    random_action.append(action)
        if(max_Q==0):
            action = initialize_action()
            action = tuple([tuple(row) for row in action])
            if(not action in Q_table[state]):
                Q_table[state].update({action:0})
                return action
            action = random_action[np.random.randint(0,len(random_action))]

        return action

# Choose which subchannel/beam to allocate
# allocate[0] = [index of subchannel device 0 allocate, subchannel device 1 allocate, subchannel device 2 allocate]
# allocate[1] = [index of beam device 0 allocate, beam device 1 allocate, beam device 2 allocate]
def allocate(action):
    sub = []  # Stores index of subchannel device will allocate
    mW = []  # Stores index of beam device will allocate
    for i in range(env.NUM_OF_DEVICE):
        sub.append(-1)
        mW.append(-1)

    avail_sub = np.full(env.NUM_OF_SUB_CHANNEL,1)
    for k in range(env.NUM_OF_DEVICE):
        if (action[k][0] == 0):
            sub[k] = action[k][1]
            avail_sub[action[k][1]] = 0
        if (action[k][0] == 1):
            mW[k] = action[k][1]
        if (action[k][0] == 2):
            mW[k] = action[k][1]
    
    for k in range(env.NUM_OF_DEVICE):
        if(action[k][0]==2):
            for i in range(len(avail_sub)):
                if(avail_sub[i]==1):
                    sub[k] = i
                    avail_sub[i] = 0
                    break

    allocate = [sub, mW]
    return allocate

# Return an matrix in which number_of_packet[k] = [number of transmit packets on sub, number of transmit packets on mW]
def compute_number_of_send_packet(action, l_sub_max, l_mW_max):
    number_of_packet = np.zeros(shape=(env.NUM_OF_DEVICE, 2))
    for k in range(env.NUM_OF_DEVICE):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if (action[k][0] == 0):
            # If l_sub_max too small, sent 1 packet and get bad reward later
            number_of_packet[k, 0] = max(1, min(l_sub_max_k, L_k))
            number_of_packet[k, 1] = 0

        if (action[k][0] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = max(1, min(l_mW_max_k, L_k))

        if (action[k][0] == 2):
            if (l_mW_max_k < L_k):
                number_of_packet[k, 1] = l_mW_max_k
                number_of_packet[k, 0] = min(l_sub_max_k, L_k - l_mW_max_k)
            if (l_mW_max_k >= L_k):
                number_of_packet[k, 0] = 1
                number_of_packet[k, 1] = L_k - 1
    return number_of_packet

# Choose power level will provide for each device over its beam/subchannel so it satisfies the power constraint base on previous rate
# Return a matrix in which power_level[k] = [power level on sub, power level on mW]
def compute_power_level(allocation):
    power_level_sub = np.zeros(shape=env.NUM_OF_DEVICE)
    power_level_mW = np.zeros(shape=env.NUM_OF_DEVICE)
    
    for i in range(env.NUM_OF_DEVICE):
        if(allocation[0][i]==-1):
            power_level_sub[i] = 0
        else:
            power_level_sub[i] = env.POWER_SET[allocation[0][i]]

        if(allocation[1][i]==-1):
            power_level_mW[i] = 0
        else:
            power_level_mW[i] = env.POWER_SET[allocation[1][i]]
    
    
    return [power_level_sub,power_level_mW]

# Return an matrix in which feedback[k] = [number of received packets on sub, number of received packets on mW]
def receive_feedback(num_of_send_packet, l_sub_max, l_mW_max):
    feedback = np.zeros(shape=(env.NUM_OF_DEVICE, 2))

    for k in range(env.NUM_OF_DEVICE):
        l_sub_k = num_of_send_packet[k, 0]
        l_mW_k = num_of_send_packet[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])

    return feedback


def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_packet_num, sent_packet_num):
    packet_loss_rate = np.zeros(shape=(env.NUM_OF_DEVICE, 2))
    for k in range(env.NUM_OF_DEVICE):
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
def compute_reward(state, power_level, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    sum = 0
    risk = 0
    for k in range(env.NUM_OF_DEVICE):
        state_k = state[k]
        sum = sum + (num_of_received_packet[k, 0] + num_of_received_packet[k, 1])/(
            num_of_send_packet[k, 0] + num_of_send_packet[k, 1]) - (1 - state_k[0]) - (1-state_k[1])
        risk_sub = risk_mW = 0
        if(state_k[0]==0 and number_of_send_packet[k,0]>0):
            risk_sub = (-power_level[0][k] + state_k[4])/(env.P_MAX - power_level[0][k])
        if(state_k[1]==0 and number_of_send_packet[k,1]>0):
            risk_mW = (-power_level[1][k] + state_k[5])/(env.P_MAX - power_level[1][k])
        risk+= risk_sub+risk_mW
    sum = ((frame_num - 1)*old_reward_value + sum)/frame_num
    return [sum, sum - risk]


# CREATE MODEL
# A Q-table is a dictionary with key=tuple(state.apend(action)), value = Q(state,action)
def initialize_Q_tables(first_state):
    Q_tables = []
    first_state = tuple([tuple(row) for row in first_state])
    for i in range(I):
        Q = {}
        add_new_state_to_table(Q,first_state)
        Q_tables.append(Q)
    return Q_tables


def add_2_Q_tables(Q1, Q2):
    q = {}
    for i in Q1:
        q.update({i:Q1[i].copy()})
    for state in Q2:
        if (state in q):
            for a in q[state]:
                if(not a in Q2[state]):
                    Q2[state].update({a:0})
                q[state][a] += Q2[state][a]
        else:
            q.update({state: Q2[state].copy()})
    return q


def average_Q_table(Q_tables):
    res = {}
    for state in range(len(Q_tables)):
        res = add_2_Q_tables(res, Q_tables[state])
    for state in res:
        for action in res[state]:
            res[state][action] = res[state][action]/I
    return res


def compute_risk_adverse_Q(Q_tables, random_Q_index):
    Q_random = Q_tables[random_Q_index].copy()
    Q_average = average_Q_table(Q_tables)
    sum_sqr = {}
    minus_Q_average = {}
    for state in Q_average:
        for action in Q_average[state]:
            Q_average[state][action] = -Q_average[state][action]
        minus_Q_average.update({state: Q_average[state].copy()})

    for i in range(I):
        sub = {}
        sub = add_2_Q_tables(sub,Q_tables[i])
        sub = add_2_Q_tables(sub, minus_Q_average)
        for state in sub:
            for action in sub[state]:
                sub[state][action] *= sub[state][action]
        sum_sqr = add_2_Q_tables(sum_sqr, sub)

    for state in sum_sqr:
        for action in sum_sqr[state]:
            sum_sqr[state][action] = -sum_sqr[state][action]*LAMBDA_P/(I-1)
    
    res = add_2_Q_tables({},sum_sqr)
    res = add_2_Q_tables(res, Q_random)
    return res


def u(x):
    return -np.exp(BETA*x)

def add_new_state_to_table(table, state):
    state = tuple([tuple(row) for row in state])
    actions = {}
    # for a in range((3*env.A**2)**env.NUM_OF_DEVICE):
    #     actions.update({a:0})
    table.update({state:actions})
    return table
    

def update_Q_table(Q_table, alpha, reward, state, action, next_state):
    next_state = tuple([tuple(row) for row in next_state])

    # Find max(Q(s(t+1),a)
    max_Q = 0
    for a in Q_table[state]:
        if(Q_table[state][a]>max_Q):
            max_Q = Q_table[state][a]

    if(not action in Q_table[state]):
        Q_table[state].update({action:0})
    Q_table[state][action] =  Q_table[state][action] + alpha[state][action] * (u(reward + GAMMA * max_Q - Q_table[state][action]) - X0)
    return Q_table


def initialize_V(first_state):
    V_tables = []
    for i in range(I):
        V = {}
        add_new_state_to_table(V,first_state)
        V_tables.append(V)
    return V_tables


def update_V(V, state, action):
    if(state in V):
        if(not action in V[state]):
            V[state].update({action:0})
        V[state][action]+=1
    else:
        add_new_state_to_table(V,state)
        V[state][action] = 1

    return V


def initialize_alpha(first_state):
    return initialize_V(first_state)


def update_alpha(alpha, V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple([tuple(row) for row in action])
    if(state in alpha):
        if(not action in alpha[state]):
            alpha[state].update({action:0})
        alpha[state][action] = 1/V[state][action]
    else:
        add_new_state_to_table(alpha,state)
        alpha[state][action] = 1/V[state][action]

    return alpha

# Set up environment
# Complex channel coefficient
def generate_h_tilde(mu, sigma, num_of_frame):
    h_tilde = []
    h_tilde_sub = env.generate_h_tilde(
        mu, sigma, num_of_frame*env.NUM_OF_DEVICE*env.NUM_OF_SUB_CHANNEL)
    h_tilde_mW = env.generate_h_tilde(
        mu, sigma, num_of_frame*env.NUM_OF_DEVICE*env.NUM_OF_BEAM)
    for frame in range(num_of_frame):
        h_tilde_sub_t = np.empty(
            shape=(env.NUM_OF_DEVICE, env.NUM_OF_SUB_CHANNEL), dtype=complex)
        for k in range(env.NUM_OF_DEVICE):
            for n in range(env.NUM_OF_SUB_CHANNEL):
                h_tilde_sub_t[k, n] = h_tilde_sub[frame*env.NUM_OF_DEVICE *
                                                  env.NUM_OF_SUB_CHANNEL + k*env.NUM_OF_SUB_CHANNEL + n]

        h_tilde_mW_t = np.empty(
            shape=(env.NUM_OF_DEVICE, env.NUM_OF_BEAM), dtype=complex)
        for k in range(env.NUM_OF_DEVICE):
            for n in range(env.NUM_OF_BEAM):
                h_tilde_mW_t[k, n] = h_tilde_mW[frame*env.NUM_OF_DEVICE *
                                                env.NUM_OF_BEAM + k*env.NUM_OF_BEAM + n]
        h_tilde_t = [h_tilde_sub_t, h_tilde_mW_t]
        h_tilde.append(h_tilde_t)
    return h_tilde

# Achievable rate
def compute_rate(device_positions, h_tilde, allocation,power_level_list,frame):
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
            p = power_level_list[0][k]
            r_sub[k] = env.r_sub(h_sub_k, device_index=k,power=p)
        if (mW_beam_index != -1):
            h_mW_k = env.compute_h_mW(device_positions, device_index=k,
                                      eta=5*np.pi/180, beta=0, h_tilde=h_tilde_mW[k, mW_beam_index],frame=frame)
            p = power_level_list[1][k]
            r_mW[k] = env.r_mW(h_mW_k, device_index=k,power=p)

    r.append(r_sub)
    r.append(r_mW)
    return r


def compute_average_r(adverage_r, last_r, frame_num):
    for k in range(env.NUM_OF_DEVICE):
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
device_positions = IO.load_positions()
# env.plot_position(ap_pos=env.AP_POSITION, device_pos=device_positions)
state = initialize_state()
action = initialize_action()
reward_first_sum = 0
allocation = allocate(action)
packet_loss_rate = np.zeros(shape=(env.NUM_OF_DEVICE, 2))
power_level = compute_power_level(allocation)
Q_tables = initialize_Q_tables(state)
V = initialize_V(state)
alpha = initialize_alpha(state)

# Generate h_tilde for all frame
h_tilde = IO.load_h_tilde()
h_tilde_t = h_tilde[0]
adverage_r = compute_rate(device_positions, h_tilde_t,
                        allocation=allocate(action),power_level_list=power_level,frame=1)
r = compute_rate(device_positions, h_tilde_t, allocation,power_level_list=power_level,frame=1)

state_plot=[]
action_plot=[]
reward_plot=[]
number_of_sent_packet_plot=[]
number_of_received_packet_plot=[]
packet_loss_rate_plot=[]
power_level_plot = []
rate_plot = []

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

for frame in range(1, env.NUM_OF_FRAME):
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
    action = choose_action(state, risk_adverse_Q)
    allocation = allocate(action)
    action_plot.append(action)
    power_level = compute_power_level(allocation)
    power_level_plot.append(power_level)
    chose_action_time += time.time()-chose_action_start_time

    # Perform action
    perform_action_start_time = time.time()
    l_max_estimate = compute_l_max(adverage_r)
    l_sub_max_estimate = l_max_estimate[0]
    l_mW_max_estimate = l_max_estimate[1]
    number_of_send_packet = compute_number_of_send_packet(
        action, l_sub_max_estimate, l_mW_max_estimate)
    number_of_sent_packet_plot.append(number_of_send_packet)
    perform_action_time += time.time()-perform_action_start_time
    

    # Get feedback
    feedback_start_time = time.time()
    r = compute_rate(device_positions, h_tilde_t, allocation,power_level,frame)
    l_max = compute_l_max(r)
    l_sub_max = l_max[0]
    l_mW_max = l_max[1]
    rate_plot.append(r)

    number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
    packet_loss_rate = compute_packet_loss_rate(
        frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
    packet_loss_rate_plot.append(packet_loss_rate)
    number_of_received_packet_plot.append(number_of_received_packet)
    adverage_r = compute_average_r(adverage_r, r, frame)
    feedback_time += time.time() - feedback_start_time

    # Compute reward
    compute_reward_start_time = time.time()
    reward_first_sum, reward_risk = compute_reward(state,power_level,number_of_send_packet,number_of_received_packet,reward_first_sum,frame)
    reward_plot.append(reward_risk)
    next_state = update_state(packet_loss_rate,number_of_received_packet,power_level)
    compute_reward_time += time.time() - compute_reward_start_time

    # Generate mask J
    J = np.random.poisson(1, I)
    state = tuple([tuple(row) for row in state])
    action = tuple([tuple(row) for row in action])
    for i in range(I):
        next_state_tuple = tuple([tuple(row) for row in next_state])
        if (J[i] == 1):
            V[i] = update_V(V[i],state,action)
            alpha[i] = update_alpha(alpha[i], V[i],state,action)
            Q_tables[i] = update_Q_table(Q_tables[i], alpha[i], reward_risk, state, action, next_state)
        if(not (next_state_tuple in Q_tables[i])):
            add_new_state_to_table(Q_tables[i], next_state_tuple)

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
IO.save(reward_first_sum,'all_reward')
IO.save(packet_loss_rate_plot,'packet_loss_rate')
IO.save(chose_action_time,'chose_action_time')
IO.save(perform_action_time,'perform_action_time')
IO.save(feedback_time,'feedback_time')
IO.save(compute_reward_time,'compute_reward_time')
IO.save(run_time,'run_time')
IO.save(rate_plot,'rate')
IO.save(power_level_plot,'power_level')