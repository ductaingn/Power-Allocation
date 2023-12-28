import numpy as np
import IO
import matplotlib.pyplot as plt
import Enviroment as env

def scatter_packet_loss_rate(device='all'):
    received = IO.load('number_of_received_packet')
    r1 = []
    r2 = []
    r3 = []
    for i in range(len(received)):
        r1.append(1-(received[i][0,0]+received[i][0,1])/6)
        r2.append(1-(received[i][1,0]+received[i][1,1])/6)
        r3.append(1-(received[i][2,0]+received[i][2,1])/6)

    x = np.arange(len(received))
    match(device):
        case 1:
            plt.scatter(x=x, y=r1)
            plt.title(f'Packet loss rate of device {device}')
        case 2:
            plt.scatter(x=x, y=r2)
            plt.title(f'Packet loss rate of device {device}')
        case 3:
            plt.scatter(x=x, y=r3)
            plt.title(f'Packet loss rate of device {device}')
        case _:
            plt.scatter(x=x, y=r1,label='Device 1')
            plt.scatter(x=x, y=r2,label='Device 2')
            plt.scatter(x=x, y=r3,label='Device 3')
            plt.title(f'Packet loss rate of devices') 
    plt.xlabel('Frame')
    plt.ylabel('Packet loss rate')
    plt.legend()
    plt.show()

def plot_reward():
    reward = IO.load('reward')
    plt.title('Reward')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.show()

def plot_position():
    ap_pos = env.AP_POSITION
    device_pos = IO.load('device_positions')
    plt.title("AP and devices Position")
    plt.scatter(ap_pos[0], ap_pos[1], color = 'b',label = 'AP')
    for i in range(len(device_pos)):
        plt.scatter(device_pos[i][0],device_pos[i][1], color = 'r',label = f"Device {i+1}")
        
    plt.xlim([0,90])
    plt.ylim([0,90])
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

def plot_power_level(device=1):
    power_level = IO.load('power_level')
    pow_sub = []
    pow_mW = []
    for i in range(len(power_level)):
        pow_sub.append(power_level[i][0][device-1])
        pow_mW.append(power_level[i][1][device-1])
    plt.plot(pow_sub,label='Power Level of Sub6-GHz')
    plt.plot(pow_mW,label='Power Level of Mm-Wave')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Power Level')
    plt.title(f'Power Level of Device {device}')
    plt.show()

def plot_interface_usage():
    action = IO.load('action')
    usage = np.zeros(shape=(env.NUM_OF_DEVICE,3))
    
    for i in range(len(action)):
        for j in range(env.NUM_OF_DEVICE):
            usage[j][action[i][j,0]]+=1
    usage = np.divide(usage,len(action)/100)
    usage = usage.transpose()
    fig,ax = plt.subplots(layout='constrained')
    x = np.arange(3)
    width = 0.2
    multiplier = 0
    interfaces = ['Sub-6GHz','mmWave','Both']
    labels = ['Device 1','Device 2','Device 3']

    for u in usage:
        offset = width*multiplier
        rects = ax.bar(x+offset,u,width)
        ax.bar_label(rects,padding=3)
        multiplier+=1
    ax.set_ylabel('Ratio [%]')
    ax.set_xticks(x+width,labels)
    ax.set_title('Interface Usage')
    plt.legend(interfaces,loc='upper right',ncols=3)
    plt.show()
                