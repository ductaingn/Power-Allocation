import Enviroment as env
import IO
import numpy as np
import matplotlib.pyplot as plt
import time

# Maximum Packet Loss Rate (PLR Requirement)

# CREAT STATE
# State is a NUM_OF_DEVICE*5 matrix
# in which state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, 
#                   ACK feedback at t-1 on sub, ACK feedback at t-1 on mW,]
def initialize_state():
    state = np.zeros(shape=(env.NUM_OF_DEVICE, 4),dtype=int)
    return state

def update_state(packet_loss_rate, feedback, rho_max):
    next_state = np.zeros(shape=(env.NUM_OF_DEVICE, 4),dtype=int)
    for k in range(env.NUM_OF_DEVICE):
        for i in range(2):
            # QoS satisfaction
            if (packet_loss_rate[k, i] <= rho_max):
                next_state[k, i] = 1
            elif (packet_loss_rate[k, i] > rho_max):
                next_state[k, i] = 0
            # Number of successfully delivered packet on each interface
            next_state[k, i+2] = feedback[k, i]
    return next_state

# CREATE ACTION
# Action is an ndarray in which action[k] = [interface_k]
def initialize_action():
    # Initialize random action at the beginning
    action = np.random.randint(0, 3, env.NUM_OF_DEVICE)
    return action

def choose_action(state, Q_table, epsilon):
    # Epsilon-Greedy
    p = np.random.rand()
    action = initialize_action()
    if (p < epsilon):
        return action
    else:
        max_Q = -np.Infinity
        state = tuple([tuple(row) for row in state])
        action = tuple(action)
        random_action = []  # Containts action with Q_value = 0
        for a in Q_table[state]:
            if(Q_table[state][a]>=max_Q):
                max_Q = Q_table[state][a]
                action = a
                if(max_Q==0):
                    random_action.append(action)
        if(max_Q==0):
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

    rand_sub = [] 
    rand_mW = []
    for i in range(env.NUM_OF_SUB_CHANNEL):
        rand_sub.append(i)
    for i in range(env.NUM_OF_BEAM):
        rand_mW.append(i)

    for k in range(env.NUM_OF_DEVICE):
        if (action[k] == 0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index)
        if (action[k] == 1):
            rand_index = np.random.randint(len(rand_mW))
            mW[k] = rand_mW[rand_index]
            rand_mW.pop(rand_index)
        if (action[k] == 2):
            rand_sub_index = np.random.randint(len(rand_sub))
            rand_mW_index = np.random.randint(len(rand_mW))

            sub[k] = rand_sub[rand_sub_index]
            mW[k] = rand_mW[rand_mW_index]

            rand_sub.pop(rand_sub_index)
            rand_mW.pop(rand_mW_index)

    allocate = [sub, mW]
    return allocate

# Return an matrix in which number_of_packet[k] = [number of transmit packets on sub, number of transmit packets on mW]
def compute_number_of_send_packet(action, l_sub_max, l_mW_max, L_k):
    number_of_packet = np.zeros(shape=(env.NUM_OF_DEVICE, 2))
    for k in range(env.NUM_OF_DEVICE):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if (action[k] == 0):
            # If l_sub_max too small, sent 1 packet and get bad reward later
            number_of_packet[k, 0] = max(1, min(l_sub_max_k, L_k))
            number_of_packet[k, 1] = 0

        if (action[k] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = max(1, min(l_mW_max_k, L_k))

        if (action[k] == 2):
            if (l_mW_max_k < L_k):
                number_of_packet[k, 1] = l_mW_max_k
                number_of_packet[k, 0] = min(l_sub_max_k, L_k - l_mW_max_k)
            if (l_mW_max_k >= L_k):
                number_of_packet[k, 0] = 1
                number_of_packet[k, 1] = L_k - 1
    return number_of_packet

# Choose power level will provide for each device over its beam/subchannel so it satisfies the power constraint base on previous rate
# Return a matrix in which power_level[k] = [power level on sub, power level on mW]
def compute_power_level(state,action,priority_coef):
    power_level_sub = np.zeros(shape=env.NUM_OF_DEVICE)
    power_level_mW = np.zeros(shape=env.NUM_OF_DEVICE)
    sub_need = 0
    sub_need_prior = []
    mw_need = 0
    mw_need_prior = []
    for k in range(env.NUM_OF_DEVICE):
        if(action[k]==0):
            if(state[k][0]==0):
                sub_need_prior.append(k)
            sub_need+=1

        elif(action[k]==1 or action[k]==2):
            if(state[k][1]==0):
                mw_need_prior.append(k)
            mw_need+=1
            if(action[k]==2):
                sub_need+=1
    
    sub_power = (priority_coef*len(sub_need_prior)+sub_need)/(priority_coef*len(sub_need_prior)+priority_coef*len(mw_need_prior)+sub_need+mw_need)*env.P_SUM
    mw_power = env.P_SUM - sub_power
    def partition(num_of_part,num_of_prior,total_p):
        if(total_p==0):
            return []
        if(num_of_prior<num_of_part):
            non_prior_power = 1/priority_coef**num_of_prior
        else:
            non_prior_power = 0
        prior_power = 1 - non_prior_power
        res = []
        for i in range(num_of_prior):
            res.append(prior_power/num_of_prior*total_p)
        for i in range(num_of_part-num_of_prior):
            res.append(non_prior_power/(num_of_part-num_of_prior)*total_p)
        return res 
    
    sub_partition = partition(sub_need,len(sub_need_prior),sub_power)
    mw_partition = partition(mw_need,len(mw_need_prior),mw_power)
    for k in range(env.NUM_OF_DEVICE):
        if(action[k]==0):
            if(state[k][2]==0):
                power_level_sub[k] = sub_partition[0]
                sub_partition.pop(0)
            else:
                power_level_sub[k] = sub_partition[-1]
                sub_partition.pop(-1)
        elif(action[k]==1 or action[k]==2):
            if(state[k][3]==0):
                power_level_mW[k] = mw_partition[0]
                mw_partition.pop(0)
            else:
                power_level_mW[k] = mw_partition[-1]
                mw_partition.pop(-1)
            if(action[k]==2):
                power_level_sub[k] = sub_partition[-1]
                sub_partition.pop(-1)

    
    return [power_level_sub,power_level_mW]

def compute_power_fix(state, action):
    power_level_sub = np.zeros(shape=env.NUM_OF_DEVICE)
    power_level_mW = np.zeros(shape=env.NUM_OF_DEVICE)

    for k in range(env.NUM_OF_DEVICE):
        if(action[k]==0):
            power_level_sub[k] = env.P_SUM/env.NUM_OF_DEVICE
        elif(action[k]==1):
            power_level_mW[k] = env.P_SUM/env.NUM_OF_DEVICE
        elif(action[k]==2):
            power_level_sub[k] = env.P_SUM/env.NUM_OF_DEVICE
            power_level_mW[k] = env.P_SUM/env.NUM_OF_DEVICE

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
        packet_loss_rate[k,0] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k,0],received_packet_num[k,0], sent_packet_num[k,0])
        packet_loss_rate[k,1] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k,1], received_packet_num[k,1], sent_packet_num[k,1])

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
        risk_sub=risk_mW=0
        if(state_k[0]==0 and num_of_send_packet[k,0]>0):
            risk_sub = env.NUM_OF_DEVICE
        if(state_k[1]==0 and num_of_send_packet[k,1]>0):
            risk_mW = env.NUM_OF_DEVICE
        risk+= risk_sub+risk_mW
    sum = ((frame_num - 1)*old_reward_value + sum)/frame_num
    return [sum, sum - risk]


# CREATE MODEL
# A Q-table is a dictionary with key=tuple(state.apend(action)), value = Q(state,action)
def initialize_Q_tables(first_state,num_Q_tables):
    Q_tables = []
    first_state = tuple([tuple(row) for row in first_state])
    for i in range(num_Q_tables):
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
                q[state][a] += Q2[state][a]
        else:
            q.update({state: Q2[state].copy()})
    return q


def average_Q_table(Q_tables,num_Q_tables):
    res = {}
    for state in range(len(Q_tables)):
        res = add_2_Q_tables(res, Q_tables[state])
    for state in res:
        for action in res[state]:
            res[state][action] = res[state][action]/num_Q_tables
    return res


def compute_risk_adverse_Q(Q_tables, random_Q_index, num_Q_tables, lambda_p):
    Q_random = Q_tables[random_Q_index].copy()
    Q_average = average_Q_table(Q_tables, num_Q_tables)
    sum_sqr = {}
    minus_Q_average = {}
    for state in Q_average:
        for action in Q_average[state]:
            Q_average[state][action] = -Q_average[state][action]
        minus_Q_average.update({state: Q_average[state].copy()})

    for i in range(num_Q_tables):
        sub = {}
        sub = add_2_Q_tables(sub,Q_tables[i])
        sub = add_2_Q_tables(sub, minus_Q_average)
        for state in sub:
            for action in sub[state]:
                sub[state][action] *= sub[state][action]
        sum_sqr = add_2_Q_tables(sum_sqr, sub)

    for state in sum_sqr:
        for action in sum_sqr[state]:
            sum_sqr[state][action] = -sum_sqr[state][action]*lambda_p/(num_Q_tables-1)
    
    res = add_2_Q_tables({},sum_sqr)
    res = add_2_Q_tables(res, Q_random)
    return res


def u(x,beta):
    return -np.exp(beta*x)

def add_new_state_to_table(table, state):
    state = tuple([tuple(row) for row in state])
    actions = {}
    def gen(actions,action,k):
        if(k==env.NUM_OF_DEVICE):
            actions.update({tuple(action.copy()):0})
            return
        for i in range(3):
            action[k]=i
            gen(actions,action,k+1)
    action = [0,0,0]
    gen(actions,action,0)
    table.update({state:actions})
    return table
    

def update_Q_table(Q_table, alpha, reward, state, action, next_state, gamma, x0, beta):
    next_state = tuple([tuple(row) for row in next_state])

    # Find max(Q(s(t+1),a)
    max_Q = 0
    for a in Q_table[state]:
        if(Q_table[state][a]>max_Q):
            max_Q = Q_table[state][a]

    Q_table[state][action] =  Q_table[state][action] + alpha[state][action] * (u(reward + gamma * max_Q - Q_table[state][action], beta) - x0)
    return Q_table


def initialize_V(first_state, num_Q_tables):
    V_tables = []
    for i in range(num_Q_tables):
        V = {}
        add_new_state_to_table(V,first_state)
        V_tables.append(V)
    return V_tables


def update_V(V, state, action):
    if(state in V):
        V[state][action]+=1
    else:
        add_new_state_to_table(V,state)
        V[state][action] = 1

    return V


def initialize_alpha(first_state, num_Q_tables):
    return initialize_V(first_state, num_Q_tables)


def update_alpha(alpha, V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    if(state in alpha):
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
    res = adverage_r.copy()
    for k in range(env.NUM_OF_DEVICE):
        res[0][k] = 1/frame_num *(last_r[0][k]+res[0][k]*(frame_num-1))
        res[1][k] = 1/frame_num *(last_r[1][k]+res[1][k]*(frame_num-1))
    return res

# l_max = r*T/d
def compute_l_max(r):
    l = np.floor(np.multiply(r, env.T/env.D))
    return l


# TRAINING
# Read from old Q-tables
def train(num_time_frame=env.NUM_OF_FRAME, 
        h_tilde=None, 
        device_positions=None, 
        priority_coef=3, 
        num_Q_tables=2,
        # Maximum Packet Loss Rate (PLR Requirement)
        RHO_MAX = 0.1,
        # L_k
        L_k = 6,
        # Risk control
        LAMBDA_P = 0.5,
        # Ultility function paremeter
        BETA = -0.5,
        # Learning parameters
        GAMMA = 0.9,
        EPSILON = 0.5,
        # Decay factor
        LAMBDA = 0.995,
        # Number of Q-tables
        I = 2,
        X0 = -1,
        power_fix=False):
    # Train with new data
    if(device_positions==None):
        device_positions = IO.load_positions()
    # env.plot_position(ap_pos=env.AP_POSITION, device_pos=device_positions)
    state = initialize_state()
    action = initialize_action()
    reward_first_sum = 0
    allocation = allocate(action)
    packet_loss_rate = np.zeros(shape=(env.NUM_OF_DEVICE, 2))
    if(power_fix==False):
        power_level = compute_power_fix(state, action)
    else:
        power_level = compute_power_level(state,action,priority_coef)
    Q_tables = initialize_Q_tables(state,num_Q_tables)
    V = initialize_V(state, num_Q_tables)
    alpha = initialize_alpha(state, num_Q_tables)

    # Generate h_tilde for all frame
    if(h_tilde==None):
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

    for frame in range(1, num_time_frame):
        # Random Q-table
        H = np.random.randint(0, num_Q_tables)
        risk_adverse_Q = compute_risk_adverse_Q(Q_tables, H, num_Q_tables, LAMBDA_P)

        # Update EPSILON
        EPSILON = EPSILON * LAMBDA

        # Set up environment
        h_tilde_t = h_tilde[frame]
        state_plot.append(state)

        # Select action
        action = choose_action(state, risk_adverse_Q, EPSILON)
        allocation = allocate(action)
        action_plot.append(action)
        if(power_fix == False):
            power_level = compute_power_fix(state,action)
        else:
            power_level = compute_power_level(state,action,priority_coef)
        power_level_plot.append(power_level)

        # Perform action
        l_max_estimate = compute_l_max(adverage_r)
        l_sub_max_estimate = l_max_estimate[0]
        l_mW_max_estimate = l_max_estimate[1]
        number_of_send_packet = compute_number_of_send_packet(
            action, l_sub_max_estimate, l_mW_max_estimate, L_k)
        number_of_sent_packet_plot.append(number_of_send_packet)
        

        # Get feedback
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

        # Compute reward
        reward_first_sum, reward_risk = compute_reward(state,power_level,number_of_send_packet,number_of_received_packet,reward_first_sum,frame)
        reward_plot.append(reward_risk)
        next_state = update_state(packet_loss_rate,number_of_received_packet, RHO_MAX)

        # Generate mask J
        J = np.random.poisson(1, num_Q_tables)
        state = tuple([tuple(row) for row in state])
        action = tuple(action)
        for i in range(num_Q_tables):
            next_state_tuple = tuple([tuple(row) for row in next_state])
            if (J[i] == 1):
                V[i] = update_V(V[i],state,action)
                alpha[i] = update_alpha(alpha[i], V[i],state,action)
                Q_tables[i] = update_Q_table(Q_tables[i], alpha[i], reward_risk, state, action, next_state, GAMMA, X0, BETA)
            if(not (next_state_tuple in Q_tables[i])):
                add_new_state_to_table(Q_tables[i], next_state_tuple)

        state = next_state

        print('frame: ',frame)
    

    return state_plot, action_plot, Q_tables, reward_plot, packet_loss_rate_plot, power_level_plot,rate_plot, number_of_sent_packet_plot, number_of_received_packet_plot
    # IO.save(number_of_received_packet_plot,'PA-number_of_received_packet')
    # IO.save(number_of_sent_packet_plot,'PA-number_of_sent_packet')
    # IO.save(reward_plot,'PA-reward')
    # IO.save(action_plot,'PA-action')
    # IO.save(state_plot,'PA-state')
    # IO.save(h_tilde,'PA-h_tilde')
    # IO.save(device_positions,'PA-device_positions')
    # IO.save(Q_tables,'PA-Q_tables')
    # IO.save(packet_loss_rate_plot,'PA-packet_loss_rate')
    # IO.save(rate_plot,'PA-rate')
    # IO.save(power_level_plot,'PA-power_level')