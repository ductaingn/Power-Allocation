import Model
import Enviroment as env
import Plot
import IO
import numpy as np
import matplotlib as plt

if __name__=="__main__":
    NUM_OF_FRAME = 10000
    h_tilde = IO.load('h_tilde')
    device_positions=None
    priority_coef=np.e
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

    state_plot, action_plot, Q_tables, reward, packet_loss_rate_plot, power_level_plot,rate_plot, number_of_sent_packet_plot, number_of_received_packet_plot = Model.train(priority_coef=np.e, h_tilde=h_tilde)

    state_plot_p_fix, action_plot_p_fix, Q_tables_p_fix, reward_p_fix, packet_loss_rate_plot_p_fix, power_level_plot_p_fix,rate_plot_p_fix, number_of_sent_packet_plot_p_fix, number_of_received_packet_plot_p_fix = Model.train(h_tilde=h_tilde,power_fix=True)

    IO.save(number_of_received_packet_plot,f'PA-number_of_received_packet')
    IO.save(number_of_sent_packet_plot,f'PA-number_of_sent_packet')
    IO.save(reward,f'PA-reward')
    IO.save(action_plot,'PA-action')
    IO.save(state_plot,'PA-state')
    IO.save(Q_tables,f'PA-Q_tables')
    IO.save(packet_loss_rate_plot,f'PA-packet_loss_rate')
    IO.save(rate_plot,f'PA-rate')
    IO.save(power_level_plot,f'PA-power_level')

    IO.save(number_of_received_packet_plot_p_fix,f'P-fix-number_of_received_packet')
    IO.save(number_of_sent_packet_plot_p_fix,f'P-fix-number_of_sent_packet')
    IO.save(reward_p_fix,f'P-fix-reward')
    IO.save(action_plot,'P-fix-action')
    IO.save(state_plot,'P-fix-state')
    IO.save(Q_tables_p_fix,f'P-fix-Q_tables')
    IO.save(packet_loss_rate_plot_p_fix,f'P-fix-packet_loss_rate')
    IO.save(rate_plot_p_fix,f'P-fix-rate')
    IO.save(power_level_plot_p_fix,f'P-fix-power_level')