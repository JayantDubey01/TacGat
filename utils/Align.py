from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd
from scipy.interpolate import make_interp_spline 

class AlignData():

    def __init__(self, capi_object, sc_object):
        
        self.capi_data = capi_object.T2   # [quaternion, translation]
        self.sc_data = sc_object.Vi    # Using Raw voltage values
        self.capi_time = capi_object.CAPITime
        self.sc_time = sc_object.tstamp2
        
        self.MissingDataList = capi_object.MissingDataList
        self.MissingDataNum = len(self.MissingDataList)
    
    def check_timezone(self):
        offset = self.capi_time[0] - self.sc_time[0]

        i = 0

        if offset > 0:
            offset = np.abs(self.capi_time[0] - self.sc_time[0])
            while(offset > 3.6e6):
                offset = np.abs(offset - (i*3.6e6))
                i = i + 1

            #print(f"Smart Cushion Timzone Offset is {i} hours behind of Polaris")   #NOTE: Put in JSON file
            self.sc_time = self.sc_time + (i*3.6e6)
        
        if offset < 0:
            offset = np.abs(self.capi_time[0] - self.sc_time[0])
            while(offset > 3.6e6):
                offset = np.abs(offset - (i*3.6e6))
                i = i + 1

            #print(f"Smart Cushion Timzone Offset is {i} hours ahead of Polaris")    #NOTE: Put in JSON file
            self.sc_time = self.sc_time - (i*3.6e6)

    # Find which device started the latest by comparing their first recorded timestamps 'start_time'. 
    # Then, find the respective index in the other device, containing the timestamp equal to, or nearest to, start_time.
    def find_start(self):

        start_time = max(self.capi_time[0],self.sc_time[0])

        if self.capi_time[0] < self.sc_time[0]: 
            #print("CAPI started earlier")   #NOTE: Put in JSON file
            start_index_capi = np.abs(self.capi_time - start_time).argmin()
            start_index_sc = 0
            
        else:
            #print("SC started earlier") #NOTE: Put in JSON file
            start_index_sc = np.abs(self.capi_time - start_time).argmin()
            start_index_capi = 0

        return start_index_capi, start_index_sc

    # Find which device ended earliest by comparing their last recorded timestamps (end_time). 
    # Then, find the respective index containing the timestamp equal to, or nearest to, end_time.
    def find_end(self):

        end_time = min(self.capi_time[-1],self.sc_time[-1])

        if self.capi_time[-1] < self.sc_time[-1]: 
            #print("CAPI ended earlier") #NOTE: Put in JSON file
            end_index_sc = np.abs(self.sc_time - end_time).argmin()
            end_index_capi = len(self.capi_time) - 1
            
        else:
            #print("SC ended earlier")   #NOTE: Put in JSON file
            end_index_capi = np.abs(self.capi_time - end_time).argmin()
            end_index_sc = len(self.sc_time) - 1
        
        return end_index_capi, end_index_sc

    def ranges(self,nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    
    # Interpolate Smart Cushion Data to the timestamps of Polaris Data. 
    # This function first finds the overlap of recordings b/w the two devices, and crops both datasets to the overlap range. 
    # Then, each column of the Smart Cushion Data is interpolated over the CAPI timestamps  
    # @NOTE: Need to interpolate Polaris Data too
    def interpret2(self, delete_missing=True):
        
        start_idx_capi, start_idx_sc = self.find_start()    # Start indices
        end_idx_capi, end_idx_sc = self.find_end() # End indices

        Nt_capi = end_idx_capi - start_idx_capi # Number of timestamps recorded by Polaris in the overlap range
        Nd = self.sc_data.shape[1]   # Number of sensors from Smart Cushion Data
        wishlist = self.capi_time[start_idx_capi:end_idx_capi]  # The time-axis sliced from the Polaris timestamps

        adjusted_sc_array = np.zeros((Nt_capi,Nd))

        # NOTE: put this in a log
        # Print how close the start and end times are from each device
        #print(f"Time dif. at start is {((np.abs(self.sc_time[start_idx_sc] - self.capi_time[start_idx_capi]))*0.001):.3f}s and end is {(np.abs(self.sc_time[end_idx_sc] - self.capi_time[end_idx_capi]) * 0.001):.3f}s")

        for i in range(Nd):
            
            xp = self.sc_time[start_idx_sc:end_idx_sc]   # timestamps
            fp = self.sc_data[start_idx_sc:end_idx_sc,i]   # data
            adjusted_sc_array[:,i] = np.interp(wishlist,xp,fp)
        
        # Crop CAPI data to the overlapped range. Also save the range of timestamps, so it can be used for plotting.
        self.adjusted_sc = torch.Tensor(adjusted_sc_array)
        self.adjusted_capi = torch.Tensor(self.capi_data[start_idx_capi:end_idx_capi,:]) 
        self.adjusted_time = self.capi_time[start_idx_capi:end_idx_capi]


        missingDataTimestamps = np.where(np.isin(self.adjusted_time, self.MissingDataList)) # Indicies of Missing Data in Timestamps column
        self.missingTimestamps = missingDataTimestamps
        
        # Convert indices to a mask: Ones everywhere except indices of the missing time data
        mask = torch.ones(self.adjusted_capi.shape[0], dtype=bool)
        mask[self.missingTimestamps] = False

        if delete_missing:
            self.adjusted_capi_clean = self.adjusted_capi[mask]
            self.adjusted_sc_clean = self.adjusted_sc[mask]
            self.adjusted_timestamps_clean = torch.Tensor(self.adjusted_time[mask])

            if self.adjusted_capi.shape[0] - self.MissingDataNum != self.adjusted_sc_clean.shape[0] or self.adjusted_sc.shape[0] - self.MissingDataNum != self.adjusted_sc_clean.shape[0]:
                raise ValueError("After removing missing data entries, the adjusted data does not match")

            # Ensure CAPi and SC data have the same number of values
            #print(f"Shape of CAPI data: {self.adjusted_capi.shape} and SC data: {self.adjusted_sc.shape}")     #NOTE: Put in JSON file   
            if self.adjusted_capi.shape[0] != self.adjusted_sc.shape[0]:
                raise ValueError("CAPI and SC data do not have the same number of values after interpolation.")
        
        # Resting state data from the 45 second mark to the 1:30 mark 5625
        self.sc_resting_state = torch.mean(self.adjusted_sc_clean[2812:5625,:],dim=0)
        self.capi_resting_state = torch.mean(self.adjusted_capi_clean[2812:5625,:],dim=0)
    
    def quatWAvg(self, Q):
        '''
        Averaging Quaternions.

        Arguments:
            Q(ndarray): an Mx4 ndarray of quaternions.
            weights(list): an M elements list, a weight for each quaternion.
        '''

        # Form the symmetric accumulator matrix
        A = torch.zeros((4, 4))
        M = Q.shape[0]
        wSum = 0

        for i in range(M):
            q = Q[i, :]
            w_i = 1.0
            A += w_i * (torch.outer(q, q)) # rank 1 update
            wSum += w_i

        # scale
        A /= wSum

        # Get the eigenvector corresponding to largest eigen value
        return np.linalg.eigh(A)[1][:, -1]
    

    def plot(self):
        # Plot the aligned pressure and motion data on the same plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

        # First subplot: Euler Angle
        ax1.plot(self.adjusted_time, self.adjusted_capi, linestyle='none', marker='o', markersize=2)
        ax1.set_ylabel('Euler Angle (Deg.)')
        ax1.set_title('Euler Angle vs Time')

        # Second subplot: Pressure
        ax2.plot(self.adjusted_time, self.adjusted_sc, linestyle='none', marker='.', markersize=2)
        ax2.set_xlabel('CAPI Time')
        ax2.set_ylabel('Pressure (g)')
        ax2.set_title('Pressure vs Time')

        plt.tight_layout()
        plt.show()