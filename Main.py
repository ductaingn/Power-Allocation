import Model
import Enviroment as env
import Plot
import IO
import numpy as np
import matplotlib as plt
import time

if __name__=="__main__":
    NUM_OF_FRAME = 10000
    # h_tilde = Model.generate_h_tilde(0,1,NUM_OF_FRAME)
    # device_positions = env.initialize_devices_pos()
    h_tilde = IO.load('PA-h_tilde')
    device_positions = IO.load('PA-device_positions')
    num_Q_tables=2
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
    I = 2
    X0 = -1
    IO.save(device_positions,f'PA-device_positions')
    IO.save(h_tilde,f'PA-h_tilde')

    for i in range(5,10):
        start = time.time()
        state_plot, action_plot, Q_tables, reward, packet_loss_rate_plot, power_level_plot,rate_plot, number_of_sent_packet_plot, number_of_received_packet_plot = Model.train(priority_coef=i,h_tilde=h_tilde, device_positions=device_positions)
        end = time.time()
        run_time = (end-start)
        print(f'Run time: {run_time} s')
        IO.save(run_time,f'PA-run_time-{i}')
    
        IO.save(number_of_received_packet_plot,f'PA-number_of_received_packet-{i}')
        IO.save(number_of_sent_packet_plot,f'PA-number_of_sent_packet-{i}')
        IO.save(reward,f'PA-reward-{i}')
        IO.save(action_plot,f'PA-action-{i}')
        IO.save(state_plot,f'PA-state-{i}')
        # IO.save(Q_tables,f'PA-Q_tables-{i}')
        IO.save(packet_loss_rate_plot,f'PA-packet_loss_rate-{i}')
        IO.save(rate_plot,f'PA-rate-{i}')
        IO.save(power_level_plot,f'PA-power_level-{i}')
