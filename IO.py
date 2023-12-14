import numpy as np
import Enviroment as env
import matplotlib
import matplotlib.pyplot as plt
import pickle

def save(param,file_name):
    name = './result/'+file_name+'.pickle'
    file = open(name,'wb')
    pickle.dump(param,file)
    print(f"Saved {file_name}!")
            
def load(file_name):
    name = './result/'+file_name+'.pickle'
    file = open(name,'rb')
    res = pickle.load(file)
    return res
