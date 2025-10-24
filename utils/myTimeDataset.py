import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

class TimeDataset(Dataset):
    def __init__(self, raw_data, slide_win, slide_stride,  t_mean=None, t_std=None, mode='train', traj=True):
        self.raw_data = raw_data
        self.slide_win = slide_win
        self.mode = mode
        self.slide_stride = slide_stride
        self.t_mean = t_mean
        self.t_std = t_std
        self.traj=traj

        self.x, self.y, self.target = self.process(raw_data)

    def __len__(self):
        return len(self.x)
    
    def process(self, all_raw_data):
        if not self.traj:
            x_list, y_list, traj_list = [], [], []

            slide_win = self.slide_win
            slide_stride = self.slide_stride

            # Split into inputs and targets
            pressure = all_raw_data[:, :11]   # shape (N, 11) PRESSURE
            transforms = all_raw_data[:, 11:18]  # shape (N, 7) TRANSFORMS 
            trajectory_full = all_raw_data[:, 18:]  # shape (N, 3) TRAJECTORY

            # normalize only translation dims
            transforms[:, 4:] = (transforms[:, 4:] - self.t_mean) / self.t_std

            total_time_len, _ = pressure.shape

            range_step = range(slide_win, total_time_len, slide_stride) if self.mode == 'train' else range(slide_win, total_time_len)

            # NOTE: NEED TO WINDOW THIS

            # Predict the next pressure mapping AND target given prior window of pressure mappings
            for i in range_step:
                signal = pressure[i-slide_win:i, :]
                transform = transforms[i, :]  
                trajectory = trajectory_full[i, :]

                x_list.append(signal)
                y_list.append(transform)
                traj_list.append(trajectory)
                
            sensor_data, transform_data, trajectory_data = torch.stack(x_list), torch.stack(y_list), torch.stack(traj_list)
            sensor_data = sensor_data.permute(0, 2, 1)

            return sensor_data, transform_data, trajectory_data
        
        else:

            x_list, y_list, traj_list = [], [], []

            slide_win = self.slide_win
            slide_stride = self.slide_stride

            # Split into inputs and targets
            pressure = all_raw_data[:, :11]   # shape (N, 11) PRESSURE
            transforms = all_raw_data[:, 11:18]  # shape (N, 7) TRANSFORMS 
            trajectory_full = all_raw_data[:, 18:]  # shape (N, 3) TRAJECTORY

            # normalize only translation dims
            transforms[:, 4:] = (transforms[:, 4:] - self.t_mean) / self.t_std

            total_time_len, _ = pressure.shape

            range_step = range(slide_win, total_time_len, slide_stride) if self.mode == 'train' else range(slide_win, total_time_len)

            # NOTE: NEED TO WINDOW THIS

            # Predict the next pressure mapping AND target given prior window of pressure mappings
            for i in range_step:
                signal = pressure[i-slide_win:i, :]
                transform = transforms[i-slide_win:i, :]  
                trajectory = trajectory_full[i-slide_win:i, :]

                x_list.append(signal)
                y_list.append(transform)
                traj_list.append(trajectory)
                
            sensor_data, transform_data, trajectory_data = torch.stack(x_list), torch.stack(y_list), torch.stack(traj_list)
            sensor_data = sensor_data.permute(0, 2, 1)

            return sensor_data, transform_data, trajectory_data         


    def __getitem__(self, index):
        return self.x[index], self.y[index], self.target[index] # Gets window of past timesteps of pressure, current timestep of transform, current position
    