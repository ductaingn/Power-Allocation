import numpy as np
import Environment as env
import matplotlib
import matplotlib.pyplot as plt
import pickle
import yaml

def save(data,file_path):
    name = file_path
    file = open(name,'wb')
    pickle.dump(data,file)
    print(f"Saved {file_path}!")
            
def load(file_path):
    name = file_path
    file = open(name,'rb')
    res = pickle.load(file)
    return res

def load_positions(process='train'):
    with open('config.yaml','rt') as file:
        config = yaml.safe_load(file)
    path = config[process]['environment']['devices_positions_path']
    return pickle.load(open(path,'rb'))

def load_h_tilde(process='train'):
    with open('config.yaml','rt') as file:
        config = yaml.safe_load(file)
    path = config[process]['environment']['h_tilde_path']
    return pickle.load(open(path,'rb'))