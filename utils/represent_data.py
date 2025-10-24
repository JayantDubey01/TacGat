import numpy as np
import torch
from scipy.spatial.transform import Rotation as R



class DataRepresentation:

    def __init__(self, sc_array, capi_array, V=None):

        self.input_data = sc_array  # [N, 11]
        self.target_data = capi_array   # [N, 7] Quaternion + translations 

        self.target_translation = self.target_data[:, 4:]
        self.target_quaternion = self.target_data[:, :4]

        
        # Find starting point of 'no motion'. This point is where all displacement data will be relative to. 
        self.low_energy_window = 300
        low_energy_idx = self.least_energy(win_len=self.low_energy_window)

        #start, end = 15000, 15500 # Good example of single plane data, but large d_
        self.start, self.end = low_energy_idx, self.target_data.shape[0]-300

        self.relative_pressure_mag = self.scviz2()
        
        # Normalize pressure mag
        self.relative_pressure_mag = (self.relative_pressure_mag - torch.mean(self.relative_pressure_mag)) / torch.std(self.relative_pressure_mag)
        self.target = self.capiviz2()

        print(f"Shape of Pressure Data: {self.relative_pressure_mag.shape}")
        print(f"Shape of Target Data: {self.target.shape}")

        self.all_data = torch.hstack((self.relative_pressure_mag, self.target))



    def scviz2(self):
        # The raw value of the sensor is relative to 0V, or infinitely large resistance, ie zero pressure or no weight. 
        # But for more meaningful displacement, we shift the reference to a some point at time t0, and for N-timepoints after that,
        # D = t_n - t_0

        input_data = self.input_data[self.start:self.end,:]

        x1 = input_data[0, :]   # Point A from least energy index
        relative_mag = input_data - x1   # Relative to the begining of time


        return relative_mag


    '''
    Args: Full T2 numpy dataset, time window parameters
    Returns: [Point A, Point B], [Trajectory], M, displacement (magnitude), delta

    Use Point A and Point B to plot the start and end, and direct line between them

    Use Trajectory and M to plot displacement lines

    Use displacement/delta to plot motion graphs against their number of points

    '''
    def capiviz2(self, cutoff=True):

        # Slice only the window of interest (Torch → NumPy)
        translation = self.target_translation[self.start:self.end, :].cpu().numpy()   # [N, 3]
                
        quat = R.from_quat(self.target_quaternion[self.start:self.end, :].cpu().numpy())  # [N, 4]

        vec = np.array([-44.32, 40.45, 50])

        vec_rotated = quat.apply(vec)                      # [N, 3] NumPy
        x0 = vec_rotated + translation                     # [N, 3] NumPy

        target_np = self.target_data[self.start:self.end, :].cpu().numpy()
        target_data = np.hstack((target_np, x0))           # [N, 10]

        return torch.tensor(target_data, dtype=torch.float32)  # convert back to Torch if needed



    def signal_cosine_sim(self, pearson=True):
        '''
        Input: Raw input and target data
        Output (H): The correlation matrix of pressure sensor displacement to pose euler-angles and magnitude in radians
        H = [N,4] where N is the number of sensors and the columns are theta_x, theta_y, theta_z, M

        As of right now, pose vector magnitude and angles are defined with respect to a randomly chosen value at the very beginning of the trial.
        It should be a value at a timestamp that is the timestamp of least displacement of pressure (constant pressure) in the beginning
        '''

        print("pressure displacement vs pose angle")
        angles_cor = self.cossim(self.theta, pearson=pearson)   # Correlation vectors

        print("pressure diplacement vs pose displacement")
        mag_cor = self.cossim(self.relative_pose_mag, pearson=pearson)

        mag_cor = mag_cor[:, np.newaxis]

        print(angles_cor.shape)
        print(mag_cor.shape)

        H = np.hstack((angles_cor, mag_cor))
        print(f"pretraining H shape: {H.shape}")
        return H

    def cossim(self, pose, pearson=False):

        if pose.ndim == 1:
            y = self.relative_pose_mag if not pearson else self.relative_pose_mag - np.mean(self.relative_pose_mag)

        else:
            y = self.theta if not pearson else self.theta - np.mean(self.theta)
        
        result = []

        # Case 1: pose is magnitude (N,)
        if y.ndim == 1:
            for sensor in self.relative_pressure_mag.T:
                x = sensor if not pearson else sensor - np.mean(sensor)
                cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                result.append(cosine)
            return np.array(result)

        # Case 2: pose is angles (N,3)
        elif y.ndim == 2:
            for sensor in self.relative_pressure_mag.T:
                x = sensor if not pearson else sensor - np.mean(sensor)
                theta = []
                for j in range(y.shape[1]):  # per angle
                    cosine = np.dot(x, y[:, j]) / (np.linalg.norm(x) * np.linalg.norm(y[:, j]))
                
                    theta.append(cosine)
                result.append(theta)
            return np.array(result)
    

    def least_energy(self, win_len=300):
        N = len(self.input_data)
        mag_squared = self.input_data ** 2
        energy_1d = mag_squared.sum(axis=1)
        # Compute sliding window energy:
        window_energy = np.convolve(energy_1d, np.ones(win_len), mode='valid')

        # Find index of smallest window
        min_win = np.argmin(window_energy)

        return min_win




    def scviz(self):
        # The raw value of the sensor is relative to 0V, or infinitely large resistance, ie zero pressure or no weight. 
        # But for more meaningful displacement, we shift the reference to a some point at time t0, and for N-timepoints after that,
        # D = t_n - t_0

        points = self.input_data[self.start:self.end,:]
        points = np.array(points)

        x1 = points[0, :]   # Point A from least energy index

        relative_mag = np.empty(points.shape)

        N = relative_mag.shape[0]

        # Compute displacement
        for i in range(N):
            relative_mag[i] = points[i] - x1   # Relative to the begining of time
            #D_list_1[i] = points[i] - x2 + x1   # Relative to the total change in pressure over the block of time

        delta_mag = relative_mag[1:] - relative_mag[:-1]
        impulse = np.trapz(delta_mag,axis=0)

        return impulse, relative_mag, delta_mag, points
    

