import numpy as np
import IO
import matplotlib.pyplot as plt
import Plot
import Environment as env
import pandas as pd

if __name__ == "__main__":
    # Plot reward
    reward = IO.load('reward')
    p = [reward[0]]
    for i in range(1,len(reward)):
        p.append(1/i*(p[i-1]*(i-1)+reward[i]))
    fig, ax = plt.subplots(2,1)
    ax[0].plot(p)
    ax[0].set_xlabel('Frame x Epoch')
    ax[0].set_ylabel('Reward')
    ax[1].set_xlabel('Frame x Epoch')
    ax[1].set_ylabel('Reward')
    ax[1].plot(p)
    ax[1].set_title('Zoom in')
    ax[1].set_ylim([3.5,4.5])
    plt.show()
    
    Plot.plot_sum_rate()
    
    # Plot epsilon
    epsilon = IO.load('epsilon')
    plt.plot(epsilon)
    plt.xlabel('Frame x Epoch')
    plt.ylabel('Epsilon')
    plt.title('Epsilon')
    plt.show()
    
    # Plot power sum packet loss rate
    Plot.plot_moving_avg_packet_loss_rate()
    # Plot power proportion
    Plot.plot_power_proportion()
    # Plot packet loss rate of each device
    Plot.plot_all_device_packet_loss_rate()
    
    # Plot model's losses
    actor_loss = IO.load('actor_loss')
    critic_loss = IO.load('critic_loss')
    actor_loss = np.array(actor_loss).flatten()
    critic_loss = np.array(critic_loss).flatten()
    fig,ax = plt.subplots(1,2)
    ax[0].plot(actor_loss)
    ax[0].set_title('Actor loss')
    ax[1].plot(critic_loss)
    ax[1].set_title('Critic loss')
    plt.show()
    
    # Plot interface usage
    Plot.plot_interface_usage_with_drop()
    