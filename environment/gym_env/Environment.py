import numpy as np
import pickle

# Number of APs
NUM_OF_AP = 1
# Number of Devices K
NUM_OF_DEVICE = 10
# Number of Sub-6Ghz channels N and mmWave beam M
NUM_OF_SUB_CHANNEL = 4
if(NUM_OF_DEVICE == 10):
    NUM_OF_SUB_CHANNEL = 16
NUM_OF_BEAM = 4 
if(NUM_OF_DEVICE == 10):
    NUM_OF_BEAM = 16
# Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10, -169/10)*1e-3
# Bandwidth Sub6-GHz = 100MHz, W_mW = 1GHz
# Bandwidth per subchannel W_sub = 100MHz/number of sub channel
W_SUB = 1e8/NUM_OF_SUB_CHANNEL
W_MW = 1e9
# Number of levels of quantitized Transmit Power
A = NUM_OF_SUB_CHANNEL
# Emitting power constraints 
P_SUM = pow(10,5/10)*1e-3*NUM_OF_DEVICE*2
# Frame Duration T_s 
T = 1e-3
# Packet size D = 8000 bit
D = 8000
# Number of frame
NUM_OF_FRAME = 10000
# LoS Path loss - mmWave
LOS_PATH_LOSS = np.random.normal(0,5.8,NUM_OF_FRAME+1)
# NLoS Path loss - mmWave
NLOS_PATH_LOSS = np.random.normal(0,8.7,NUM_OF_FRAME+1) 

# initialize position of AP.
# the AP was located at the central of the area
# the position of each AP is the constant
AP_POSITION = (0, 0)

# the function calculates the distance to the nearest AP
def distance_to_AP(pos_of_device):
    distance = np.sqrt((pos_of_device[0] - AP_POSITION[0])**2 + (pos_of_device[1] - AP_POSITION[1])**2)
    return distance


# initialize device's postion with random value
# after initializing any device's position, check the distance from that device to the nearest AP,
# if the distance is satisfied, store it into the array list_of_devices.
def generate_devices_positions(num_of_device=NUM_OF_DEVICE):
    device_positions = [
        [0, 20],     # Device 1
        [20, 0],     # Device 2 (Blocked by obstacle)
        [-85, -80],  # Device 3
        [-45, 40],   # Device 4
        [10, -70],   # Device 5
        [-40, -20],  # Device 6 (Blocked by obstacle)
        [-40, 15],   # Device 7
        [60, 55],    # Device 8
        [45, 5],     # Device 9
        [50, -40],   # Device 10
        [40, 60],    # Device 11 (Blocked by obstacle)
        [-20, -60],  # Device 12
        [-20, 80],   # Device 13
        [20, -40],   # Device 14 (Blocked by obstacle)
        [-80, 80]    # Device 15        
    ]
        
    return device_positions[:num_of_device]

# Path loss model
def path_loss_sub(distance):
    return 38.5 + 30*(np.log10(distance))


def path_loss_mW_los(distance,frame):
    X = LOS_PATH_LOSS[frame]
    return 61.4 + 20*(np.log10(distance))+X


def path_loss_mW_nlos(distance,frame):
    X = NLOS_PATH_LOSS[frame]
    return 72 + 29.2*(np.log10(distance))+X


# Main Transmit Beam Gain G_b
def G(eta=5*np.pi/180, beta=0, epsilon=0.1):
    return (2*np.pi-(2*np.pi-eta)*epsilon)/(eta)

def generate_h_tilde_device_chanel(mu:float, sigma:float, amount:int)->np.ndarray:
    """
    Channel coefficient
        h=h_tilde* 10^(-pathloss/20)
        h_tilde = (a + b*i)/sqrt(2)
    
        in which a and b is random value from a Normal(mu, sigma) distribution
    """
    re = np.random.normal(mu, sigma, amount)
    im = np.random.normal(mu, sigma, amount)
    h_tilde = []
    for i in range(amount):
        h_tilde.append(complex(re[i], im[i])/np.sqrt(2))
    return np.array(h_tilde)

def generate_h_tilde(num_timestep:int, num_device:int, num_subchannel:int, num_beam:int, mu:float=0, sigma:float=1, save:bool=True, save_path:str='environment/scenario_1/h_tilde.pickle')->np.ndarray:
    h_tilde = []
    for k in range(num_device):
        h_tilde_k_sub, h_tilde_k_mw = [], []
        for n in range(num_subchannel):
            h_tilde_k_sub.append(generate_h_tilde_device_chanel(mu, sigma, num_timestep))
            
        for m in range(num_beam):
            h_tilde_k_mw.append(generate_h_tilde_device_chanel(mu, sigma, num_timestep))

        h_tilde.append(np.array([h_tilde_k_sub, h_tilde_k_mw]))

    h_tilde = np.array(h_tilde) # [device index, interface, channel index, timestep]
    h_tilde = np.transpose(h_tilde, (3, 1, 0, 2)) # [timestep, interface, device index, channel_index] (3, 1, 0, 2)

    if save:
        with open(save_path, 'wb') as file:
            pickle.dump(h_tilde, file)

    return h_tilde


def compute_h_sub(list_of_devices, device_index, h_tilde):
    h = np.abs(h_tilde* pow(10, -path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0))**2
    return h


def compute_h_mW(list_of_devices, device_index, h_tilde, frame,eta=5*np.pi/180, beta=0):
    h = 0
    # device blocked by obstacle
    if (device_index in [1,5,10,13]):
        path_loss = path_loss_mW_nlos(distance_to_AP(list_of_devices[device_index]),frame)
        h = np.abs(G(eta, beta)*h_tilde* pow(10, -path_loss/20)*0.01)**2  # G_Rx^k=epsilon
    # device not blocked
    else:
        path_loss = path_loss_mW_los(distance_to_AP(list_of_devices[device_index]),frame)
        h = np.abs((G(eta, beta)**2)*h_tilde * pow(10, -path_loss/20))**2  # G_Rx^k = G_b

    return h

# gamma_sub(h,k,n) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on subchannel n with channel coefficient h
def gamma_sub(h, p=P_SUM):
    power = h*p
    interference_plus_noise = W_SUB*SIGMA_SQR
    # for b in range(NUM_OF_AP):
    #     if(b!=AP_index):
    #         interference_plus_noise += pow(abs(h[device_index,channel_index]),2)*P
    return power/interference_plus_noise

# gamma_mW(k,m) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on beam m with channel coeffiction h


def gamma_mW(h, p=P_SUM):
    power = h*p
    interference_plus_noise = W_MW*SIGMA_SQR
    # for b in range(NUM_OF_AP):
    #     if(b!=AP_index):
    #         interference_plus_noise += pow(abs(h[device_index,beam_index]),2)*P
    return power/interference_plus_noise


# achievable data rate r_bkf (t) for the link between
# AP b, device k and for application f using bandwidth Wf at scheduling frame t
def r_sub(h, power):
    return W_SUB*np.log2(1+gamma_sub(h,p=power))


def r_mW(h, power):
    return W_MW*np.log2(1+gamma_mW(h,p=power))


def packet_loss_rate(t, old_packet_loss_rate, omega_kv, l_kv):
    if (l_kv == 0):
        packet_loss_rate = ((t-1)/t)*old_packet_loss_rate
        return packet_loss_rate
    elif (l_kv > 0):
        packet_loss_rate = (1/t)*((t-1)*old_packet_loss_rate + (1-omega_kv/l_kv))
        return packet_loss_rate
