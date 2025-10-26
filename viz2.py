import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader, ConcatDataset
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.MatchIOFiles import matching_io
from utils.represent_data import DataRepresentation
from utils.myTimeDataset import TimeDataset

from run_GCTCN import train_GCTCN
from gt_setuprun import setup_and_run
import glob

def datasaver(path):
    io_pair = np.array(matching_io(path + "/sc", path + "/capi"))

    # --- load each trial, compute its own H, and pack (x,y,Hrep) ---
    for i in range(1):
    #for i in range(2):
        CAPIreader = ReadCapi2(io_pair[i, 0])
        SCreader   = ReadSC2(io_pair[i, 1])

        Synced = AlignData(CAPIreader, SCreader)
        Synced.check_timezone()
        Synced.interpret2(delete_missing=True)

        sc = Synced.adjusted_sc_clean    # [N, 11]
        capi = Synced.adjusted_capi_clean  # [N, 7]
    
        torch.save({"sc": sc, "capi": capi}, f"{i}.pt")

def main(path):


    #path = "/home/jayantdubey/Desktop/AllData"
    #datasaver(path)
    #exit(-1)
    
    #num_channels = [11, 88, 88, 66, 66, 44, 44, 11] 
    #num_channels = [11, 22, 11]
    num_channels = [11, 32, 64, 32, 32, 16, 11]

    traj = False

    r = setup_and_run(train=True, slide_win=20,slide_stride=10,
                      lr=1e-4, graph_name='gg',num_channels=num_channels,epochnum=15,kernel_size=2,traj=traj,pathtest=path)
    r.setup_data(batch_size=16)
    r.run(traj=traj)
    
    #r = setup_and_run(train=False, slide_win=20,slide_stride=10,
     #                 lr=1e-4, graph_name='gg',num_channels=num_channels,epochnum=15,kernel_size=2,traj=traj)
    
    print("Finished")


    '''
    WITH THIS SETUP:

    EULER MAE: [2.2474232 1.1995281 2.0827358] mean: 1.8432289361953735 std: 0.4601040482521057
    std per axis: [1.4002022 1.3885952 1.5377879]
    
    TRANSLATION MAE: [4.9180026  0.60234034 0.7660494 ] mean: 2.095463991165161 std: 1.9969547986984253
    std per axis: [0.8358833  0.7954281  0.48849934] BestSoFar_SL10_SS5
    '''


def test(path):


    num_channels = [11, 32, 64, 32, 32, 16, 11]
    # --- load each trial for TimeDataset ---

    num_channels = [11, 32, 64, 32, 32, 16, 11]

    traj = False
    PT = ".pt"
    r = setup_and_run(train=False, slide_win=20,slide_stride=10,
                      lr=1e-4, graph_name='gg',num_channels=num_channels,epochnum=14,kernel_size=2,traj=traj,pathtest=path)
    r.setup_data(batch_size=16)
    r.run(traj=traj)

path = "TacGat/TEST4.pt"
#main(path)
#test(path=path)

data = ReadSC2("TacGat/datasets/SC2024-10-24-data002.sc")


'''
    num_channels = [11, 22, 66, 33, 11] 
    
    r = setup_and_run(train=True, slide_win=50,slide_stride=1,
                      lr=1e-4, graph_name='g',num_channels=num_channels,epochnum=15,kernel_size=4)
    r.setup_data(batch_size=64) STABLEDRAFT1


        num_channels = [11, 22, 66, 88, 66, 33, 11] 
    #num_channels = [11, 32, 64, 32, 32, 16, 11]


    r = setup_and_run(train=True, slide_win=128,slide_stride=1,
                      lr=1e-4, graph_name='g',num_channels=num_channels,epochnum=15,kernel_size=4)
    r.setup_data(batch_size=64) STABLE_6DEG


        
    num_channels = [11, 88, 88, 66, 66, 44, 44, 11] 
    #num_channels = [11, 32, 64, 32, 32, 16, 11]


    r = setup_and_run(train=True, slide_win=164,slide_stride=2,
                      lr=1e-4, graph_name='g',num_channels=num_channels,epochnum=40,kernel_size=4)
    r.setup_data(batch_size=64)
    r.run() STABLE_BetterForm

    

'''
{'count': 2965, 'mean': 81.174072265625, 'std': 65.38034057617188, 'median': 35.966278076171875, 'p75': 172.53720092773438, 
 'p90': 175.72435607910157, 'p95': 177.63959045410155, 'p99': 179.40102355957032, 'max': 179.91844177246094}
''''''
