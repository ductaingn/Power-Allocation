import Enviroment as env
import IO
import numpy as np

actions = [] # Contains all combinations of action
device = [] # Contains combinations of (interface, power level) of one device
action_device = [0,0]
sub_channel = np.full(env.NUM_OF_SUB_CHANNEL,0)
mW_beam = np.full(env.NUM_OF_BEAM,0)

def gen_one_device(device,action,k):
    if(k==2):
        device.append(action.copy())
        return
    if(k==0):
        for i in range(3):
            action[k]=i
            gen_one_device(device,action,k+1)
    else:
        # number of beam = number of subchannel
        for i in range(env.NUM_OF_BEAM):
            action[k]=i
            gen_one_device(device,action,k+1)

gen_one_device(device,action_device,0)
allocated = np.full(len(device),0)
def mark_allocated(i):
    interface = device[i][0]
    if(interface == 0):
        allocated[i] = 1
        return
    else:
        beam = device[i][1]
        for j in range(len(device)):
            if(device[j][1] == beam):
                allocated[j] = 1
        return
    
def mark_unallocated(i):
    interface = device[i][0]
    if(interface == 0):
        allocated[i] = 0
        return
    else:
        beam = device[i][1]
        for j in range(len(device)):
            if(device[j][1] == beam):
                allocated[j] = 0
        return
    
action = [0,0,0]
def gen(actions,action,k):
    if(k==env.NUM_OF_DEVICE):
        # actions.append(tuple([tuple(row) for row in  action.copy()]))
        actions.append(action.copy())
        return
    for i in range(len(device)):
        if(allocated[i]==0):
            action[k] = device[i]
            mark_allocated(i)
            gen(actions,action,k+1)
            mark_unallocated(i)
        else:
            continue
gen(actions,action,0)
IO.save(actions,'possible_actions')