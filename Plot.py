import numpy as np
import IO
import matplotlib.pyplot as plt
import Enviroment as env

def scatter_packet_loss_rate(device='all'):
    received = IO.load('number_of_received_packet')

    x = np.arange(len(received))

    if(device!= 'all'):
        plr = []
        for i in range(len(received)):
            plr.append(1-(received[i][device-1,0]+received[i][device-1,1])/6)
            
        plt.scatter(x=x, y=plr[device-1])
        plt.title(f'Packet loss rate of device {device}')

    else:
        plr = np.zeros(shape=(env.NUM_OF_DEVICE,len(received)))
        for i in range(len(received)):
            for k in range(env.NUM_OF_DEVICE):
                plr[k,i]=(1-(received[i][k,0]+received[i][k,1])/6)

        for k in range(env.NUM_OF_DEVICE):
            plt.scatter(x=x,y=plr[k],label = f'Device {k+1}')
        plt.title(f'Packet loss rate of all devices') 

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
    plt.scatter(ap_pos[0], ap_pos[1], color = 'r',marker = 's',label = 'AP')
    for i in range(len(device_pos)):
        if(i == 1 or i == 5):
            plt.scatter(1/2*device_pos[i][0]+1/2*ap_pos[0],1/2*device_pos[i][1]+1/2*ap_pos[1], color = 'black',label = 'Obstacle',marker = 'd')
        plt.scatter(device_pos[i][0],device_pos[i][1], color = 'b')
        plt.text(device_pos[i][0]-0.4,device_pos[i][1]+0.8,f"D{i+1}",fontsize=12)
        
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

def plot_action(device=1):
    action = IO.load('action')
    plot = []
    for i in range(len(action)):
        plot.append(action[i][device-1,0])

    plt.plot(plot)
    plt.title(f'Action of device {plot}')
    plt.xlabel('Frame')
    plt.ylabel('Action')
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
    x = np.arange(env.NUM_OF_DEVICE)
    width = 0.2
    multiplier = 0
    interfaces = ['Sub-6GHz','mmWave','Both']
    labels = [f'Device {i+1}' for i in range(env.NUM_OF_DEVICE)]

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
                
def plot_rate(device=1):
    rate = IO.load('rate')
    sub = []
    mW = []
    avg_sub = []
    avg_mW = []
    for i in range(len(rate)):
        sub.append(rate[i][0][device-1])
        mW.append(rate[i][1][device-1])
        avg_sub.append(np.mean(sub[0:i]))
        avg_mW.append(np.mean(mW[0:i]))
    
    plt.plot(avg_sub,label='Rate over Sub6-GHz')
    plt.plot(avg_mW,label='Rate over mmWave')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.title(f'Average rate of device {device}')
    plt.legend()
    plt.show()