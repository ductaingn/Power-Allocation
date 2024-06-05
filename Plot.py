import numpy as np
import IO
import matplotlib.pyplot as plt
import Enviroment as env

def plot_packet_loss_rate(device=1):
    # received = IO.load('number_of_received_packet')
    # sent = IO.load('number_of_sent_packet')
    # device-=1
    # received_device_sub, received_device_mW = 0,0
    # sent_device_sub, sent_device_mW = 0,0
    # plrsub = []
    # plrmw = []
    # for i in range(len(sent)):
    #     received_device_sub += received[i][device][0]
    #     received_device_mW += received[i][device][1]

    #     sent_device_sub += sent[i][device][0]
    #     sent_device_mW += sent[i][device][1]
        
    #     plrsub.append(1-received_device_sub/sent_device_sub)
    #     plrmw.append(1-received_device_mW/sent_device_mW)

    # plt.plot(plrsub,label='sub')
    # plt.plot(plrmw,label='mw')
    # plt.legend()
    # device+=1
    # plt.title(f'Packet loss rate of device {device}')
    # plt.show()

    received = IO.load('number_of_received_packet')
    sent = IO.load('number_of_sent_packet')
    device-=1
    received_device = 0
    sent_device = 0
    plr = []
    for i in range(len(sent)):
        received_device += received[i][device][0] + received[i][device][1]

        sent_device += sent[i][device][0] + sent[i][device][1]
        
        plr.append(1-received_device/sent_device)

    plt.plot(plr)
    device+=1
    plt.title(f'Packet loss rate of device {device}')
    plt.show()
        
def plot_moving_avg_packet_loss_rate():
    received = IO.load('number_of_received_packet')
    sent = IO.load('number_of_sent_packet')
    plr = []
    for i in range(len(sent)):
        r = 0
        s = 0
        for device in range(env.NUM_OF_DEVICE):
            r+= received[i][device][0]+received[i][device][1]
            s+= sent[i][device][0]+sent[i][device][1]
        plr.append(1-r/s)
    p = []
    for i in range(len(plr)):
        p.append(np.mean(plr[0:i]))
    plt.plot(p,label='plr')
    plt.legend()
    plt.title('Moving average Sum Packet loss rate of all devices')
    plt.show()

def scatter_packet_loss_rate(device='all'):
    received = IO.load('number_of_received_packet')
    sent = IO.load('number_of_sent_packet')

    x = np.arange(len(received))

    if(device!= 'all'):
        plr = []
        for i in range(len(received)):
            plr.append(1-(received[i][device-1,0]+received[i][device-1,1])/(sent[i][device-1,0]+sent[i][device-1,1]))
            
        plt.scatter(x=x, y=plr)
        plt.title(f'Packet loss rate of device {device}')

    else:
        plr = np.zeros(shape=(env.NUM_OF_DEVICE,len(received)))
        for i in range(len(received)):
            for k in range(env.NUM_OF_DEVICE):
                plr[k,i]=(1-(received[i][k,0]+received[i][k,1])/(sent[i][k,0]+sent[i][k,1]))

        for k in range(env.NUM_OF_DEVICE):
            plt.scatter(x=x,y=plr[k],label = f'Device {k+1}')
        plt.title(f'Packet loss rate of all devices') 

    plt.xlabel('Frame')
    plt.ylabel('Packet loss rate')
    plt.legend()
    plt.show()

def plot_reward():
    reward = IO.load('PA-reward')
    p = []
    for i in range(len(reward)):
        p.append(np.mean(reward[0:i]))
    plt.title('Moving average Reward-PA')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.plot(p)
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
        
    plt.xlim([-env.width/2,env.width/2])
    plt.ylim([-env.length/2,env.length/2])
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

def plot_power_level(device=1):
    count_sub = np.zeros(shape=env.A)
    count_mW = np.zeros(shape=env.A)
    power_level = IO.load('power_level')
    pow_sub = []
    pow_mW = []
    for i in range(len(power_level)):
        pow_sub.append(power_level[i][0][device-1])
        pow_mW.append(power_level[i][1][device-1])
        count_sub[pow_sub[i]]+=1
        count_mW[pow_mW[i]]+=1
    x = np.arange(len(power_level))
    plt.scatter(x=x,y=pow_sub,label='Power Level of Sub6-GHz')
    plt.scatter(x=x,y=pow_mW,label='Power Level of Mm-Wave')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Power Level')
    plt.title(f'Power Level of Device {device}')
    plt.show()
    mean_sub = 0
    mean_mW = 0
    for i in range(env.A):
        mean_sub += i*count_sub[i]
        mean_mW += i*count_mW[i]
    print(f'Mean Level of sub: ',mean_sub/count_sub.sum())
    print(f'Mean Level of mW: ',mean_mW/count_mW.sum())

def scatter_action(device=1):
    action = IO.load('action')
    plot = []
    for i in range(len(action)):
        plot.append(action[i][device-1][0])

    fig,ax = plt.subplots()
    x = np.arange(len(action))
    ax.scatter(x,y=plot)
    ax.set_title(f'Action of device {device}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Action')
    plt.show()

def plot_interface_usage():
    action = IO.load('PA-action')
    usage = np.zeros(shape=(env.NUM_OF_DEVICE,3))
    
    for i in range(len(action)):
        for j in range(env.NUM_OF_DEVICE):
            usage[j][action[i][j]]+=1
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
    for i in range(len(rate)):
        if(i==0):
            sub.append(rate[i][0][device-1])
            mW.append(rate[i][1][device-1])

        else:
            sub.append(1/i*((i-1)*sub[i-1] + rate[i][0][device-1]))
            mW.append(1/i*((i-1)*mW[i-1] + rate[i][1][device-1]))

    plt.plot(sub,label='Rate over Sub6-GHz')
    plt.plot(mW,label='Rate over mmWave')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.title(f'Average rate of device {device}')
    plt.legend()
    plt.show()

def plot_sum_rate():
    rate = IO.load('rate')
    sub = []
    mW = []
    avg_sub = []
    avg_mW = []
    for i in range(len(rate)):
        sumsub = 0
        summW = 0
        for k in range(env.NUM_OF_DEVICE):
            sumsub+= rate[i][0][k]
            summW+= rate[i][1][k]
        sumsub/=env.NUM_OF_DEVICE
        summW/=env.NUM_OF_DEVICE

        if(i==0):
            sub.append(sumsub)
            mW.append(summW)
        else:
            sub.append(1/i* (sub[i-1]*(i-1) + sumsub))
            mW.append(1/i* (mW[i-1]*(i-1) + summW))

    plt.plot(sub,label = 'Rate over Sub6-GHz')
    plt.plot(mW,label='Rate over mmWave')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.title(f'Average sum rate of all device-PA')
    plt.legend()
    plt.show()

def plot_powerlevel_distribution(interface):
    count = np.zeros(env.A)
    action = IO.load('action')
    fix,ax = plt.subplots()

    if (interface=='sub'):
        for i in range(len(action)):
            for k in range(env.NUM_OF_DEVICE):
                count[action[i][k][0]]+=1

        x = np.arange(env.A)
        for i in range(env.A):
            ax.bar(x=x[i],height=count[i],label = f'{i}')

    elif(interface=='mW'):
        for i in range(len(action)):
            for k in range(env.NUM_OF_DEVICE):
                count[action[i][k][1]]+=1

        x = np.arange(env.A)
        for i in range(env.A):
            ax.bar(x=x[i],height=count[i],label = f'{i}')

    plt.xlabel('Power level')
    ax.set_title(f'Power level distribution of {interface}')
    ax.legend(title='Level ')
    plt.show()
    mean = 0
    for i in range(env.A):
        mean += i*count[i]
    print(f'Mean Level of {interface}: ',mean/count.sum())

def plot_moving_avg_powerlevel(device=1):
    count_sub = np.zeros(shape=env.A)
    count_mW = np.zeros(shape=env.A)
    power_level = IO.load('power_level')
    action = IO.load('PA-action')
    pow_sub = []
    avg_sub = []
    pow_mW = []
    avg_mW = []
    for i in range(len(power_level)):
        if(action[i][device-1]==0):
            pow_sub.append(power_level[i][0][device-1])
        if(action[i][device-1]==1):
            pow_mW.append(power_level[i][1][device-1])
        else:
            pow_sub.append(power_level[i][0][device-1])
            pow_mW.append(power_level[i][1][device-1])
        avg_sub.append(np.mean(pow_sub[0:len(pow_sub)-1]))
        avg_mW.append(np.mean(pow_mW[0:len(pow_mW)-1]))
        # count_sub[pow_sub[i]]+=1
        # count_mW[pow_mW[i]]+=1
    x = np.arange(len(power_level))
    plt.plot(avg_sub,label='Power Level of Sub6-GHz')
    plt.plot(avg_mW,label='Power Level of Mm-Wave')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Power Level')
    plt.title(f'Moving Average Power Level of Device {device}')
    plt.show()
    # mean_sub = 0
    # mean_mW = 0
    # for i in range(env.A):
    #     mean_sub += i*count_sub[i]
    #     mean_mW += i*count_mW[i]
    # print(f'Mean Level of sub: ',mean_sub/count_sub.sum())
    # print(f'Mean Level of mW: ',mean_mW/count_mW.sum())

def bench_mark():
    chose_action_time = IO.load('chose_action_time')
    compute_reward_time = IO.load('compute_reward_time')
    feedback_time = IO.load('feedback_time')
    perform_action_time = IO.load('perform_action_time')
    update_Q_time = IO.load('update_Q_time')
    run_time = IO.load('run_time')
    others = run_time - chose_action_time - compute_reward_time - feedback_time - perform_action_time - update_Q_time
    chart =[chose_action_time*100/run_time,
            compute_reward_time*100/run_time,
            feedback_time*100/run_time,
            perform_action_time*100/run_time,
            update_Q_time*100/run_time,
            others*100/run_time]
    labels = ['Choose action time','Compute reward time','Feedback time','Perform action time','Update Q-tables time','Others']
    plt.pie(chart,startangle=90)
    plt.axis('equal')
    plt.legend(loc='right',labels=labels)
    plt.show()

def plot_power_proportion():
    action = np.array(IO.load('action'))
    action = action.reshape((len(action),2,10))
    fig, ax = plt.subplots(2,5)
    for i in range(env.NUM_OF_DEVICE):
        power_device_sub,power_device_mW = action[:,:,i][:,0], action[:,:,i][:,1]
        p_sub, p_mw = [power_device_sub[0]],[power_device_mW[0]]
        for j in range(1,len(power_device_sub)):
            p_sub.append(1/j*(p_sub[-1]*(j-1)+power_device_sub[j]))
            p_mw.append(1/j*(p_mw[-1]*(j-1)+power_device_mW[j]))

        ax_col = i%5
        ax_row = int(i/5)
        ax[ax_row,ax_col].plot(p_sub,label='Sub6-GHz')
        ax[ax_row,ax_col].plot(p_mw,label='mmWave')
        ax[ax_row,ax_col].legend()
        ax[ax_row,ax_col].set_title(f'Device {i+1}')
        ax[ax_row,ax_col].set_xlabel('Frame x Epoch')
        ax[ax_row,ax_col].set_ylabel('Power proportion')
        ax[ax_row,ax_col].set_ylim([-0.01,0.15])

    fig.suptitle('Power proportion')
    plt.show()

def plot_all_device_packet_loss_rate():
    received = IO.load('number_of_received_packet')
    sent = IO.load('number_of_sent_packet')

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(2,5,figsize=(2240*px,1400*px))
    for k in range(env.NUM_OF_DEVICE):
        ax_col = k%5
        ax_row = int(k/5)

        received_device = 0
        sent_device = 0
        plr = []
        for i in range(len(sent)):
            received_device += received[i][k][0] + received[i][k][1]
            sent_device += sent[i][k][0] + sent[i][k][1]
            
            plr.append(1-received_device/sent_device)

        ax[ax_row,ax_col].plot(plr,label='PLR')
        ax[ax_row,ax_col].set_xlabel('Frame x Epoch')
        ax[ax_row,ax_col].set_ylabel('Packet loss rate')
        ax[ax_row,ax_col].set_title(f'Device {k+1}')
        ax[ax_row,ax_col].set_ylim([-0.02,1.02])
        ax[ax_row,ax_col].axhline(y=0.1,color='red',linestyle='--',label='$\\rho_{max}$')
        ax[ax_row,ax_col].legend()

    fig.suptitle('Moving avg. Packet loss rate')
    plt.savefig('test.pdf')
    plt.show()