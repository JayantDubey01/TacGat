import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity 
from utils.rot6d import RotationTransformer

            
class weighted_corrnn(nn.Module):
    def __init__(self, S: torch.Tensor, hidden=80):
        """
        S = [N, 11] N = 1 for now, and 11 pressure displacement at time N 
        Output = [theta_x, theta_y, theta_z, M]
        """
        in_dim = S.shape[-1]
        corr_dim = 16
        Y = 7
    

        super().__init__()

        # Shared feature encoder
        #Works pretty well, but has spikes and DC offset
        
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

        self.branch_theta_y = nn.Sequential(
            nn.Linear(in_dim, 16),
        )

        self.branch_theta_z = nn.Sequential(
            nn.Linear(in_dim, 16),
        )

        self.branch_theta_x = nn.Sequential(
            nn.Linear(in_dim, 16),
        )

        self.final_output = nn.Sequential(
            nn.Linear(3*corr_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, Y)
        )

        self.mse_loss = nn.MSELoss()
        self.tf = RotationTransformer('euler_angles', 'matrix', from_convention='YZX')


        self.clamp_param = nn.Parameter(torch.tensor([0.9, 0.9, 0.9]))

        self.register_buffer("clamp_min", torch.tensor([0.7, 0.7, 0.7]))
        self.register_buffer("clamp_max", torch.tensor([1.1, 1.1, 1.1]))

        
    def forward(self, x, H):
        # H: [sensors, k] or [B, sensors, k] (k≥3). We'll broadcast if needed.
        if H.ndim == 2:
            H = H.unsqueeze(0).expand(x.size(0), -1, -1)   # [B, sensors, k]

        sensors = H.shape[1]
        if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == sensors:
            x = x.squeeze(1)
        assert x.ndim == 2 and x.shape[1] == sensors, f"Expected x [B,{sensors}], got {x.shape}"

        x = self.backbone(x)

        # Your function should handle [B, sensors, k] + [B, sensors]
        H_set = self.covariance_weighted_by_dimension(H, x)  # expect [B, sensors, ≥3]
        
        # Turn off H
        #H_set = x.unsqueeze(-1).repeat(1,1,3)

        y_weight = H_set[..., 0] * x
        z_weight = H_set[..., 1] * x
        x_weight = H_set[..., 2] * x

        y_output = self.branch_theta_y(y_weight)
        z_output = self.branch_theta_z(z_weight)
        x_output = self.branch_theta_x(x_weight)

        fused = torch.cat([x_output, y_output, z_output], dim=1)  # [B, 12] if corr_dim=4
        return self.final_output(fused)  # [B, 7]

    def loss(self, q_pred, q_true, t_pred, t_true):
        Lq = self.quat_loss(q_pred, q_true)
        Lt = self.trans_loss(t_pred, t_true)
        return Lq + Lt

    def quat_loss(self, q_pred, q_true):
        q_pred = F.normalize(q_pred, dim=-1)
        q_true = F.normalize(q_true, dim=-1)
        dot = torch.sum(q_pred * q_true, dim=-1).abs()
        return (1 - dot).mean()
    
    def trans_loss(self, t_pred, t_true):
        return F.mse_loss(t_pred, t_true)
        
    def apply_euler_transform(self, theta, M=None):
        # theta: [N,3], M: [N,1]
        matrix_pred = self.tf.forward(theta)   # [N,3,3]
        unit_vec = torch.tensor(self.unit_v,
                                device=theta.device,
                                dtype=theta.dtype)
        rotated = torch.matmul(matrix_pred, unit_vec)  # [N,3]

        if M is not None:
            # ensure correct shape for broadcasting
            if M.dim() == 1:  # [N]
                M = M.unsqueeze(1)  # [N,1]
            
            vec = rotated * M
        else:
            return rotated
    
    def covariance_weighted_by_dimension(self, H, x):
        """
        H: [B, N_sensors, N_outputs]
        x: [B, N_sensors]
        return: [B, N_sensors, N_outputs]
        """
        # expand x to: [B, N_sensors, 1], then broadcast multiply
        x_exp = x.unsqueeze(-1)  # [B, 11, 1]
        return x_exp * H         # [B, 11, 4]