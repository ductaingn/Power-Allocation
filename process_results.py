import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_interface_usage(history_df:pd.DataFrame, num_devices:int, title:str = None, save_path:str = None):
    """
    Plot the interface usage from the DataFrame.
    Args:
        history_df (pd.DataFrame): DataFrame containing the history of interface usage.
        num_devices (int): Number of devices in the environment.
        title (str, optional): Title of the plot. Defaults to None.
        save_path (str, optional): Path to save the plot. If None, the plot will not be saved. Defaults to None.
    """
    num_received_packet = np.zeros((num_devices, 2))
    num_droped_packet = np.zeros((num_devices, 2))
    interfaces = ['Sub6GHz', 'mmWave']
    for k in range(num_devices):
        for i, iface in enumerate(interfaces):
            num_received_packet[k, i] = history_df[f'Device {k+1}/ Num. Received packet/ {iface}'][:-1].sum()
            num_droped_packet[k, i] = history_df[f'Device {k+1}/ Num. Droped packet/ {iface}'][:-1].sum()

    fig, ax = plt.subplots(figsize=(16, 9))

    x = np.arange(num_devices)
    bar_width = 0.35

    # Sub6 bars
    ax.bar(x - bar_width/2, num_received_packet[:,0], bar_width, label='Thành công trên Sub-6GHz', color='blue', hatch='/', edgecolor='k')
    ax.bar(x - bar_width/2, num_droped_packet[:,0], bar_width, bottom=num_received_packet[:,0], label='Không thành công', color='red')

    # mmWave bars
    ax.bar(x + bar_width/2, num_received_packet[:,1], bar_width, label='Thành công trên mmWave', color='green', hatch='\\', edgecolor='k')
    ax.bar(x + bar_width/2, num_droped_packet[:,1], bar_width, bottom=num_received_packet[:,1], label='Không thành công', color='red')

    # Labels and formatting
    ax.set_xlabel('Thiết bị')
    ax.set_ylabel('Số gói tin')
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{k+1}' for k in range(num_devices)])
    ax.set_ylim(0, 70_000)

    # Custom legend for interfaces
    sub6_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', hatch='/', edgecolor='k')
    mmwave_patch = plt.Rectangle((0, 0), 1, 1, fc='green', hatch='\\', edgecolor='k')
    drop_patch = plt.Rectangle((0, 0), 1, 1, fc='red')
    ax.legend(
        [sub6_patch, mmwave_patch, drop_patch],
        ['Gói tin gửi thành công trên Sub-6GHz', 'Gói tin gửi thành công trên mmWave', 'Gói tin mất mát'],
        bbox_to_anchor=(0., 1.02, 1., .102), 
        loc='lower left',
        ncols=1, 
        mode="expand", 
        borderaxespad=0.,
        fontsize=32  # Set legend font size here
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Plot saved to {save_path}")

    plt.show()

def get_result_table(history_dfs:list[pd.DataFrame], num_devices:int) -> pd.DataFrame:
    """
    Generate a summary table from the history DataFrame.
    Args:
        history_df (list[pd.DataFrame]): List of DataFrames containing the history of runs.
        num_devices (int): Number of devices in the environment.
    Returns:
        pd.DataFrame: Summary table with average success rate, reward, and packet loss rates.
    """
    result_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
            [
                ("", "Algorithm"),
                ("", "Reward"),
                ("", "Avg. Success"),
            ]+
            [
                ("PLR",f"D{i+1}") for i in range(num_devices)
            ]+
            [("", "Δ̄ρ")]
        )
    )

    for history_df in history_dfs:
        algorithm = history_df["Algorithm"].iloc[0]
        reward = history_df["Overall/ Reward"].iloc[-2]
        avg_suc = 1.0 - history_df["Overall/ Sum Packet loss rate"].iloc[-2]
        plr = [
            history_df[f"Device {i+1}/ Packet loss rate/ Global"].iloc[-2].item() for i in range(num_devices)
        ]
        delta_rho = sum([0.1 - plr[i] for i in range(num_devices)])/num_devices
        result_df.loc[len(result_df)] = [algorithm, reward, avg_suc] + plr + [delta_rho]

    return result_df