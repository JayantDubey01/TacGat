from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch_geometric.data import Data

def get_feature_map():
    """ Hardcoded feature list """
    feature_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] 
    return feature_list

def get_tactile_graph_struc(feature_list):
    """ Generates a graph structure from adjacency matrix for tactile features. """
    adjacency_matrix = pd.read_csv('/home/jayantdubey/workspace/NeuralNet/Tactile-GAT/data/tactile/my_graph_struc.csv', index_col='Name')
    struc_map = {ft: adjacency_matrix.columns[adjacency_matrix.loc[ft] == 1].tolist() for ft in feature_list}
    return struc_map
    

def build_loc_net(struc, feature_map):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in feature_map:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in feature_map:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
    
    return torch.tensor(edge_indexes, dtype=torch.long)

def get_fc_graph_struc(feature_list):
    """ Generates a fully connected graph structure. """
    return {ft: [other_ft for other_ft in feature_list if other_ft != ft] for ft in feature_list}


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """
    Adjusts edge indices for batch processing in graph neural networks.

    Parameters:
    - org_edge_index (torch.Tensor): Original edge indices (2, edge_num).
    - batch_num (int): Number of batches.
    - node_num (int): Number of nodes per batch.

    Returns: 
    - torch.Tensor: Adjusted edge indices for the batches. 
    """

    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]  
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num
        # Example of 32 batches @ i = 3: [:, 3*11:4*11] += 3*11-> [:, 33:44] += 33 
    return batch_edge_index.long()