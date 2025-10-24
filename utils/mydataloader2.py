import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as gDataset
from torch_geometric.data import Data as gData
import torchvision

#import cv2
import matplotlib.pyplot as plt
import numpy as np
import roma
from scipy.spatial.transform import Rotation as R
import splines

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.MatchIOFiles import matching_io

import os

def construct_data(io_pair, config):

    def consecTimeIndex(timesteps):
        timesteps = np.array(timesteps).flatten()
        diff = timesteps[1:] - timesteps[:-1]
        consecutive = []
        
        idx = -1
        while idx + 1 < len(diff):
            idx = idx + 1
            count = 0
            flag = 1
            if diff[idx] == 1:
                while flag:
                    count = count + 1

                    if idx + count > len(diff)-1:
                        break

                    if diff[idx+count] == 1:
                        pass
                    else:
                        flag = 0
            end = 0
            if count > 1:
                end = idx+count-1
                consecutive.append((idx,end))
                idx = end - 1
            else: 
                continue            

        return consecutive

    sc_file = io_pair[1]
    capi_file = io_pair[0]

    CAPIreader = ReadCapi2(capi_file)
    SCreader = ReadSC2(sc_file)    

    # Sync smart cushion data to polaris data, returns tensorized data. CAPI data is nx11, and SC data is nx4. Its a quaternion. 
    Synced = AlignData(CAPIreader,SCreader)
    Synced.check_timezone()
    Synced.interpret2(delete_missing=True)
    missingDataIndices = Synced.missingTimestamps


    input_data = Synced.adjusted_sc_clean
    target_data = Synced.adjusted_capi_clean
    timestamps = Synced.adjusted_timestamps_clean

    missing_ranges = consecTimeIndex(missingDataIndices)
    missing_ranges = np.array(missing_ranges).flatten()
    missingDataIndices =  np.array(missingDataIndices).flatten()

    # Normalize translation by subtracting mean
    translations = target_data[:, 4:]
    translation_avg = torch.mean(translations, axis=0)
    norm_translations = torch.subtract(translations, translation_avg)
    target_data[:, 4:7] = norm_translations*0.001   # Normalize to meters instead of mm. This will help with numerical instability

    # Go through missing timestamps, if there are more than 2 consecutive timesteps then break the dataset into subsets
    if missing_ranges.any(): 
        
        subsets = []

        N = missing_ranges.shape[0]
        x = np.arange(0,N+1,1)
        x = x[1:]
        x[-1] = -1; 
        x = x.reshape(int(x.shape[0]/2),2)
        i = missing_ranges[0]

        inp = input_data[:missingDataIndices[i]-1,:]
        tar = target_data[:missingDataIndices[i]-1,:]
        data = torch.concatenate((inp.t(), tar.t()), dim=0)

        subsets.append(data)
        
        for entry in x:

            if entry[1] == -1:  # handle the last case
                i = missing_ranges[entry[0]]
                t = missingDataIndices[i]

                inp = input_data[t+2:, :]
                tar = target_data[t+2:, :]
                data = torch.concatenate((inp.t(), tar.t()), dim=0)

                subsets.append(data)
            else:
                i = missing_ranges[entry[0]]
                i_next = missing_ranges[entry[1]]

                t = missingDataIndices[i] + 2
                t_next = missingDataIndices[i_next] - 1

                inp = input_data[t:t_next, :]
                tar = target_data[t:t_next, :]
                data = torch.concatenate((inp.t(), tar.t()), dim=0)

                subsets.append(data)         

        for idx, entry in enumerate(subsets):   # Replace with 'slide_win' from config
            if entry.shape[1] < config['slide_win']:
                subsets.pop(idx)

        return subsets
        

    else:
        # First transpose then, concatenate input and targets = [18, timesteps]
        subsets = []
        data = torch.concatenate((input_data.t(),target_data.t()),dim=0)
        subsets.append(data)

        #print(f"Dataset Shape: {data.shape}")


        return subsets

class CustomDataset(Dataset):
    def __init__(self, input_file, target_file, io=None, multiple_datasets=False, learn_delta=False, 
                 learn_angular_velocity = False, target_norm=True, input_norm=None, target_transform=None, windowed=None):
        
        sc_file = input_file  # File of SC data
        capi_file = target_file    # File of Polaris data

        # Read Raw Data from one trial into respective class objects
        CAPIreader = ReadCapi2(capi_file)
        SCreader = ReadSC2(sc_file)

        #print(f"Unit Vector U: {CAPIreader.avg_unit_u}")
        self.unit_u = CAPIreader.avg_unit_u
        
        
        # Sync smart cushion data to polaris data, returns tensorized data. CAPI data is nx11, and SC data is nx4. Its a quaternion. 
        Synced = AlignData(CAPIreader,SCreader)
        Synced.check_timezone()
        Synced.interpret2(delete_missing=True)
        self.missingDataIndices = Synced.missingTimestamps

        # Dataset transformations
        self.target_norm = target_norm
        self.input_norm = input_norm
        self.learn_delta = learn_delta
        self.learn_angular_velocity = learn_angular_velocity
        self.target_transform = target_transform

        # Initialize Tensor Input/Target datasets from one trial
        self.input_data = Synced.adjusted_sc_clean
        self.target_data = Synced.adjusted_capi_clean
        self.target_data_test = Synced.adjusted_capi_clean
        self.timestamps = Synced.adjusted_timestamps_clean

        # Should have some sense of how long the trial was, and then from there cut o

        print(f"Original Input Shape: {self.input_data.shape}")
        print(f"Original Target Shape: {self.target_data.shape}")

        self.rest_state_pressure_map = Synced.sc_resting_state
        self.rest_state_pose = Synced.capi_resting_state

        #print(f"Resting state pose: {self.rest_state_pose}")
        
        if self.target_norm:
            '''
            Subtracts data with mean for a stationairy dataset
            '''
            translations = self.target_data[:, 4:]
            translation_avg = torch.mean(translations, axis=0)
            norm_translations = torch.subtract(translations, translation_avg)
            self.target_data[:, 4:7] = norm_translations
            #self.input_data = self.input_transform(self.input_data) # Save as heatmap images, then read from new directory
        
        if self.input_norm:
            avg_pressure_map = torch.mean(self.input_data,axis=0)
            norm_maps = torch.subtract(self.input_data,avg_pressure_map)
            self.input_data = norm_maps
        
        if self.learn_delta:
            '''
            This reduces the target data to the quaternion difference between two consecutive timepoints. In other words, the delta between a timestep and the 
            timestep before is computed. The same delta is computed with the pressure map. 
            '''

            #self.input_data = torch.concatenate((self.input_data[1:],self.input_data[:-1]),dim=1)  # What if the difference is also some kind of SO3 object?
            self.input_data = self.input_data[1:] - self.input_data[:-1]
            new_target = torch.empty(self.target_data.shape[0]-1,self.target_data.shape[1],dtype=torch.float32)           
            
            for i in range(self.target_data.shape[0]-1):

                diff_t = self.target_data[i+1,4:] - self.target_data[i,4:] 
                diff_q = self.compute_quat_dif(self.target_data[i+1,:4], self.target_data[i,:4])
                new_target[i,:4] = torch.Tensor(diff_q)
                new_target[i,4:] = torch.Tensor(diff_t)
            
            self.target_data = new_target

            print(f"Delta Input Shape: {self.input_data.shape}")
            print(f"Delta Target Shape: {self.target_data.shape}")
                
        # SO3 Conversion
        if self.target_transform:
            self.translation_data = self.target_data[:,4:7]   # Extract translation vector
            self.rotation_data = self.quatToSO3(self.target_data[:,0:4])   # Convert Quaternion to SO3 Object
            
            translation = self.translation_data.unsqueeze(2)
            target_transform = torch.cat((self.rotation_data, translation), dim=2)

            padding = torch.Tensor([0,0,0,1]).unsqueeze(0).unsqueeze(0).expand(self.rotation_data.shape[0],1,4)
            self.target_data = torch.cat((target_transform, padding), dim=1) 

        # Target_data now can either be 1 continuous dataset, or multiple subdatasets depending on the nature of the missing values
        # If it has to be broken down into multiple subdatasets, then the windowing should window each subset, concatenate back into a single dataset
        # and the index of where the breaks are should be stored. This will be used for the modified __getitem__ func where we will need to translate a 
        # global index into one of the subsets. 

        # Clean and then window the clean dataset(s)
        if windowed:
            print("Windowing")
            self.win_len = 30
            #self.input_data = [self.input_data[i:i+self.win_len] for i in range(len(self.input_data) - self.win_len + 1)]
            #self.target_data = [self.target_data[i:i+self.window_len] for i in range(len(self.target_data) - self.window_len + 1)]
            self.input_data = self.input_data.unfold(0,self.win_len,1)
            self.target_data = self.target_data.unfold(0,self.win_len,1)
            list_of_healthy_windows = []

            for idx, window in enumerate(self.target_data):
                if not self.hasTooManyNans(window):
                    list_of_healthy_windows.append(idx)
                else:
                    pass
            
            self.input_data = self.input_data[list_of_healthy_windows]
            self.target_data = self.target_data[list_of_healthy_windows]

            print(f"Window Input: {self.input_data.shape}")
            print(f"Window Target: {self.target_data.shape}")
      
        
    def __len__(self):
        return len(self.input_data)    # Number of samples
    
    # this is reading
    def __getitem__(self, idx):

        return self.input_data[idx], self.target_data[idx] 

    def hasTooManyNans(self, data: torch.Tensor):
        nan_mask = torch.all(data.isnan(), dim=1)
        unique_consecutive_values, counts = torch.unique_consecutive(nan_mask, return_counts=True)
        start_index = 0
        true_segments = []

        for i, val in enumerate(unique_consecutive_values):
            current_count = counts[i]
            if val:  # If the consecutive segment is 'True'
                end_index = start_index + current_count - 1
                true_segments.append((start_index, end_index))
            start_index += current_count
        
        for segment in true_segments:
            if (segment[1] - segment[0] + 1) >= 5:
                return True
        
        return False
    
    def compute_quat_dif(self, q1, q2):

        q1 = R.from_quat(q1)
        q2 = R.from_quat(q2)

        q2_inv = q2.inv()
        q2_from_q1 = q1 * q2_inv

        return q2_from_q1.as_quat()

    def angular_velocities(self, q1, q2, dt):
        return (2 / dt) * np.array([
            q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
            q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
            q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])


    def quatToSO3(self, target_data):
        # Convert quaternion data to Symmetric Orthogonal Rotation Matrices. Append translation vector to create a 4x4 homogenous transform matrix
        rotation_matrices = self.quaternion_to_rotmat(target_data)

        # Convert to symmetric orthogonal matrices
        return self.RotM_to_SO3(rotation_matrices)
        # Type hints: inp: expects a Tensor, -> output expects a tensor output
    
    def quaternion_to_rotmat(self, inp: torch.Tensor, **kwargs) -> torch.Tensor:
        # without normalization
        # normalize first
        x = inp.reshape(-1, 4)
        x_norm = x.norm(dim=1, keepdim=True)
        x = x / x_norm
        return roma.unitquat_to_rotmat(x)

    # Cleans rotation matrices into SO3 Objects. By multiplying the U and Vt after SVD and ensuring the determinant to equal 1.
    # det(A) = 1 means that the volume preserved under transformation.  
    def RotM_to_SO3(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

        x: should have size [batch_size, 9]

        Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
        """
        m = x.view(-1, 3, 3)
        u, s, v = torch.linalg.svd(m)
        vt = v.transpose(1,2)

        det = torch.linalg.det(torch.matmul(u, vt))
        det = det.view(-1, 1, 1)
        vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)

        r = torch.matmul(u, vt)

        return r  

    

'''
capi_path = "/home/jayantdubey/Desktop/CAPIdata"
sc_path = "/home/jayantdubey/Desktop/SCdata"

input_output_pairs = matching_io(sc_path,capi_path)  # List of matching input/output file paths

target, input = input_output_pairs[2]

#initial_dataset = CustomDataset(input, target, target_norm=True,target_transform=True)
initial_dataset = construct_data(input_output_pairs[0])
'''