import numpy as np
import Enviroment as env
import matplotlib
import matplotlib.pyplot as plt
import pickle

def save(data,file_name):
    name = './result/'+file_name+'.pickle'
    file = open(name,'wb')
    pickle.dump(data,file)
    print(f"Saved {file_name}!")
            
def load(file_name):
    name = './result/'+file_name+'.pickle'
    file = open(name,'rb')
    res = pickle.load(file)
    return res

def load_positions():
    return pickle.load(open('/home/nguyen/Projects/Group ICN/Topic 2/Source Code/result/device_positions.pickle','rb'))

def load_h_tilde():
    return pickle.load(open('/home/nguyen/Projects/Group ICN/Topic 2/Source Code/result/h_tilde.pickle','rb'))