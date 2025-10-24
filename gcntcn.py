import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch_geometric.nn.conv import MessagePassing
from torch.nn.utils import weight_norm
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU, Dropout, Embedding
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from models.myTactileGAT import TactileGAT
import torch.nn.functional as F
from utils.rot6d import RotationTransformer

from myutils import get_batch_edge_index
import math

''' -------------------------------- TEMPORAL CONVOLUTION LAYER -------------------------------------------'''

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # assumes x is [B, C, L + padding]; trims 'padding' from the right to keep causal shape
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    '''
    Input -> Convolution -> Slice -> ReLU -> Dropout -> Convolution -> Slice -> ReLU -> Dropout -> out
    out -> (downsample if needed) -> ReLU -> result
    '''
    def __init__(self, num_sensors, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()


        self.conv1 = weight_norm(nn.Conv1d(num_sensors, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.max_pool_layer = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(num_sensors, n_outputs, 1) if num_sensors != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [B, C_in, L]
        out = self.net(x)  # [B, C_out, L]
        res = x if self.downsample is None else self.downsample(x)  # [B, C_out, L]
        return self.relu(out + res)  # [B, C_out, L]

class TemporalConvNet(nn.Module):
    '''
    Input = [B, num_sensors, win_len]
    Output = [B, C_out, win_len]  (length-preserving via padding+chomp)
    '''
    def __init__(self, num_sensors, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_sensors if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, num_sensors, L]
        return self.network(x)  # [B, num_channels[-1], L]

class TCNLayer(nn.Module):
    """
    Temporal Convolutional Network (Bai et al. 2018).
    """
    def __init__(
        self,
        num_sensors=None,
        num_channels=None,
        n_filters=30,
        kernel_size=2,
        drop_prob=0.1,
    ):
        super().__init__()
        self.tcn_channel_dim = num_sensors
        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "fc.bias": "final_layer.fc.bias",
        }

        self.tcnblock = TemporalConvNet(num_sensors, num_channels, kernel_size=kernel_size, dropout=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_sensors, time_win]
        returns: [B, C_out, time_win] where C_out = num_channels[-1]
        """
        # Ensure expected 3D input
        assert x.dim() == 3, f"TCNLayer expects [B, num_sensors, T], got shape {tuple(x.shape)}"
        out = self.tcnblock(x)  # [B, C_out, T]
        return out

''' ---------------------------------SPATIAL GRAPH CONVOLTUION -------------------------------------------'''

class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, feat_out, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, feat_out)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(feat_out)

    def forward(self, x, edge_index):
        # Conv 1
        h = self.conv1(x, edge_index)
        h = self.norm1(h)            # normalize features
        h = torch.relu(h)
        h = self.dropout(h)          # dropout AFTER activation

        # Conv 2
        g = self.conv2(h, edge_index)
        g = self.norm2(g)            # normalize again
        g = torch.relu(g)            # final feature
        g = self.dropout(g)

        return g  # [num_nodes_total, feat_out]


class ResidualGCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, feat_out, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, feat_out)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(feat_out)

        self.dropout = nn.Dropout(dropout)

        # If in_dim != out_dim, use projection to match sizes for residual
        self.proj = nn.Linear(in_channels, feat_out) if in_channels != feat_out else nn.Identity()

    def forward(self, x, edge_index):
        identity = self.proj(x)

        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = torch.relu(h)
        h = self.dropout(h)

        g = self.conv2(h, edge_index)
        g = self.norm2(g)

        # Residual connection + activation
        out = torch.relu(g + identity)
        return out
    
class GCNHead(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = AttnPool(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, gcn_out):  # gcn_out: [B, N, H]
        x = self.norm(gcn_out)
        x = self.pool(x)              # [B, H]
        #x = torch.tanh(self.fc(x))    # [B, out_dim], bounded
        return x

class AttnPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):             # x: [B, N, H]
        w = self.attn(x)              # [B, N, 1]
        a = torch.softmax(w, dim=1)   # [B, N, 1]
        pooled = (a * x).sum(dim=1)   # [B, H]
        return pooled


class GCNTCN(nn.Module):
    def __init__(self, input, edge_index_sets, gcn_hidden_dim, gcn_out_dim, num_channels,kernel_size):
        super(GCNTCN, self).__init__()

        # TCN layer inputs
        self.num_sensors = input.shape[1]         # node_num 
        self.num_channels = num_channels  # last value (11) to maintain sensor dimension for GCN
        self.kernel_size = kernel_size

        # GCN layer inputs
        self.edge_index_sets = edge_index_sets
        self.win_len = input.shape[2]             # time window length (C_feat)
        self.hidden_channels = 48
        self.out_dim = gcn_out_dim # 7

        # Out heads
        self.out_layer_dim = 6  # 6d rotation

        tcn_layer_num = 1
        gcn_layer_num = len(self.edge_index_sets)

        # Tactile GAT
        self.TacGat = TactileGAT(self.edge_index_sets, self.num_sensors, 
                                 out_layer_inter_dim=self.hidden_channels, input_dim=self.win_len)

        self.tcn_layers = nn.ModuleList([
            TCNLayer(self.num_sensors, self.num_channels, kernel_size=self.kernel_size, drop_prob=0.3)
            for _ in range(tcn_layer_num)
        ])
        # GCN in_channels == time length (C_feat)
        self.gcn_layers = nn.ModuleList([
            ResidualGCNLayer(in_channels=self.win_len, hidden_channels=self.hidden_channels, feat_out=self.out_dim)
            for _ in range(gcn_layer_num)
        ])

        self.gcnhead = GCNHead(hidden_dim=self.out_dim, out_dim=self.out_dim)


        print(f"MODEL PARAMS: ksize: {kernel_size},\nwin_len: {self.win_len},\nhide_chan: {self.hidden_channels}\n,gcn out dim:{self.out_dim}\n,finaloutdim:{self.out_layer_dim}\n")

        self.cache_edge_index_sets = [None] * len(edge_index_sets)
        self.init_params()

        # Final head: take graph-level embedding [B, out_dim] -> [B, 7]
        self.final_output_traj = nn.Sequential(
            nn.Linear(4*self.out_dim, self.hidden_channels *2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_channels *2, 6*self.win_len)  # Outputs flattened 
        )

        # Final head: take graph-level embedding [B, out_dim] -> [B, 7]
        self.final_output = nn.Sequential(
            nn.Linear(4*self.out_dim, self.hidden_channels *2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_channels *2, 7)  # 4quat + 3t 
        )

        '''        self.final_output = nn.Sequential(
            nn.Linear(4*self.out_dim, self.hidden_channels *2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear( self.hidden_channels *2, self.hidden_channels *2),
            nn.ReLU(),
            nn.Dropout(0.1),

            
            nn.Linear(self.hidden_channels *2, 7)  # 4quat + 3t 
        )'''

        self.quatFmat = RotationTransformer('quaternion','matrix')
        self.rotFmat = RotationTransformer('rotation_6d','matrix')
        self.eps = 1e-6

    def prepare_batch_edge_index(self, i, edge_index, batch_num):
        # Builds batched edge_index with per-batch node offsets
        if (self.cache_edge_index_sets[i] is None or
            self.cache_edge_index_sets[i].size(1) != edge_index.size(1) * batch_num):
            self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, self.num_sensors).to(self.device)
        return self.cache_edge_index_sets[i]

    def forward(self, data):
        # data: [B, num_sensors, time]
        input, self.device = data.clone(), data.device

        batch_num = input.size(0)     # B
        num_nodes = self.num_sensors  # N
        time_len = input.size(2)      # T (== self.win_len)

        # ----- TCN -----
        x = input
        for tcn in self.tcn_layers:
            x = tcn(x)  # [B, 11, T]

        if torch.isnan(x).any():
            raise RuntimeError("TCN output NaNs")

        # flatten for GCN input
        x_gcn = x.reshape(batch_num * num_nodes, time_len)  # [B*N, T]

        gcn_outputs = []
        # ---- Multi-GCN feature extraction ----
        for i, edge_index in enumerate(self.edge_index_sets):
            edge_index = self.edge_index_sets[min(i, len(self.edge_index_sets)-1)]
            batch_edge_index = self.prepare_batch_edge_index(i, edge_index, batch_num)

            x_gcn_out = self.gcn_layers[i](x_gcn, batch_edge_index)  # [B*N, F]

            # stabilize each GCN head
            #x_gcn_out = self.norm_layers[i](x_gcn_out)  # LayerNorm
            x_gcn_out = torch.tanh(x_gcn_out)           # squash to [-1, 1]

            gcn_outputs.append(x_gcn_out)

        # ---- Concat features from all GCNs ----
        x_cat = torch.cat(gcn_outputs, dim=1)          # [B*N, F_total]

        # ---- Reshape back to [B, N, F_total] for pooling ----
        x_cat = x_cat.view(batch_num, num_nodes, -1)   # [B, N, F_total]

        x_mean = x_cat.mean(dim=1)             # [B, H]
        x_max, _ = x_cat.max(dim=1)            # [B, H]
        x_graph = torch.cat([x_mean, x_max], dim=-1)  # [B, 2H]

        if torch.isnan(x_cat).any():
            raise RuntimeError("GCN output NaNs")

        # ---- Use your GCNHead (Norm → AttnPool → Tanh(FC)) ----
        #x_graph = self.gcnhead(x_cat)                  # [B, out_dim_from_head]


        # ---- Final MLP head ----
        x_out = self.final_output(x_graph)             # [B, final_out_dim]
        #x_out = self.final_output_traj(x_graph)             # [B, final_out_dim]
        
        return x_out
    
    '''
    Need to convert quaternion to loss
    '''


    def loss(self, s_pred, q_true, trajectory):
    
        '''
        INPUT: Predicted unnormalized quaternion, True normalized quaternion
        OUTPUT: 
        '''
        
        s_pred_quat = s_pred[:, :4]
        s_pred_trans = s_pred[:, 4:]

        s_pred_quat = torch.nn.functional.normalize(s_pred_quat, p=2, dim=-1)
        #q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)

        s_pred_quat = s_pred_quat * torch.sign((s_pred_quat * q_true).sum(dim=-1, keepdim=True))

        mat_pred = self.quatFmat.forward(s_pred_quat)
        mat_true = self.quatFmat.forward(q_true)
    
        if torch.isnan(mat_pred).any():
            raise RuntimeError("pred output NaNs")
    
        if torch.isnan(mat_true).any():
            raise RuntimeError("true output NaNs")

        mat_true_transpose = torch.transpose(mat_true, 1, 2)

        p = mat_pred @ mat_true_transpose
        trace = p.diagonal(dim1=1, dim2=2).sum(dim=1)
        cos = (trace - 1) / 2

        cos = torch.clamp(cos, -1 + self.eps, 1 - self.eps)
        #loss = (1.0 - cos).pow(2)   # More sensitive to small angles
        theta_loss = (1.0 - cos)   # Less sensitive to small angles


        # Angle of rotation
        theta = torch.acos(cos)  # radians

        if torch.isnan(theta).any():
            raise RuntimeError("Loss output NaNs")
        
        trans_loss = self.trans_loss(s_pred_trans, trajectory)

        loss = trans_loss + theta_loss

        return loss.mean()

    def loss_traj(self, s_pred, r_true, trajectory, unit_vector): # s_pred is now N*6

        # Shapes the output to [B, i-win:i, 6]
        s_pred = s_pred.contiguous().view(-1, self.win_len, self.out_layer_dim)
        B = s_pred.shape[0]
        unit_vec_expanded = unit_vector.view(1, 1, 3).expand(B, self.win_len, 3)

        s_pred_flat = s_pred.reshape(B * self.win_len, 6)   # [B*win_len, 6]

        # Convert to batch wise rotation matrices
        mat_pred = self.rotFmat.forward(s_pred_flat)
        mat_pred_V = mat_pred.view(B, self.win_len, 3, 3)   # [B, win_len, 3, 3]


        unit_vec_rotated = torch.matmul(
            unit_vec_expanded.unsqueeze(2),   # [B, win_len, 1, 3]
            mat_pred_V                          # [B, win_len, 3, 3]
        ).squeeze(2)                          # [B, win_len, 3]

        #pred_vector = unit_vec_rotated + trajectory # Still just rotation otuput 

        #loss_transX = torch.abs(trajectory[:, 0] - pred_vector[:, 0]) / self.win_len
        #loss_transY = torch.abs(trajectory[:, 1] - pred_vector[:, 1]) / self.win_len
        #loss_transZ = torch.abs(trajectory[:, 2] - pred_vector[:, 2]) / self.win_len

        #sum_trans_loss = loss_transX + loss_transY + loss_transZ

        if torch.isnan(mat_pred_V).any():
            raise RuntimeError("pred output NaNs")

        if r_true.ndim == 2 and r_true.shape[-1] == 4:
            # [B*win_len, 4] → [B, win_len, 4]
            r_true = r_true.view(B, self.win_len, 4)
        elif r_true.ndim == 3 and r_true.shape[-1] == 4:
            # [B, win_len, 4] → ensure dims match
            assert r_true.shape[0] == B and r_true.shape[1] == self.win_len, \
                f"r_true has shape {r_true.shape}, expected [B={B}, win_len={self.win_len}, 4]"
        else:
            raise ValueError(f"r_true has shape {r_true.shape}, expected [..., 4]")

        # Flatten and (optionally) normalize quaternion before mat conversion
        r_true_flat = r_true.reshape(B * self.win_len, 4)           # [B*win_len, 4]
        r_true_flat = torch.nn.functional.normalize(r_true_flat, p=2, dim=-1, eps=1e-8)

        mat_true = self.quatFmat.forward(r_true_flat)              # [B*win_len, 3, 3]
        #mat_true_V = mat_true.view(B, self.win_len, 3, 3)

        cos = self.kracher_mean(mat_pred, mat_true).clamp(-1+self.eps, 1-self.eps)
        angle = torch.acos(cos)               # radians, in [0, π]
        return angle.mean()
        

    def trans_loss(self, t_pred, t_true):
        return F.mse_loss(t_pred, t_true)
    
    

    def kracher_mean(self, pred_mat, true_mat):
        def rotmat_to_axis_angle(mat):
            # trace → angle
            trace = mat.diagonal(dim1=1, dim2=2).sum(dim=1)
            trace = trace.clamp(-1 + self.eps, 1 - self.eps)
            cos_theta = (trace - 1) / 2
            theta = torch.acos(cos_theta).unsqueeze(1)  # [B,1]

            # axis from skew-symmetric part
            axis = torch.stack([
                mat[:, 2, 1] - mat[:, 1, 2],
                mat[:, 0, 2] - mat[:, 2, 0],
                mat[:, 1, 0] - mat[:, 0, 1],
            ], dim=1)  # [B,3]

            # unit axis (pred_vec = axis * theta)
            sin_theta = torch.sin(theta).clamp(min=self.eps)
            vec = axis / (2 * sin_theta)  # rotation vector (unit * magnitude=theta)

            return vec  # [B,3]

        # vectorized conversion
        v_pred = rotmat_to_axis_angle(pred_mat)   # [B,3]
        v_true = rotmat_to_axis_angle(true_mat)   # [B,3]

        # cosine similarity between rows
        dot = (v_pred * v_true).sum(dim=1, keepdim=True)  # [B,1]
        pred_norm = v_pred.norm(dim=1, keepdim=True)      # [B,1]
        true_norm = v_true.norm(dim=1, keepdim=True)      # [B,1]

        cos_sim = dot / (pred_norm * true_norm + self.eps)
        return cos_sim


    def rot_error_deg_batched(self, R_pred, R_true, eps=1e-6):

        # Case 1: prediction is flattened and must be reshaped and converted
        if R_pred.ndim == 2:      # shape [B,6] → rotation matrices
            B = R_pred.shape[0]
            R_pred = R_pred.view(B, self.out_layer_dim)     # [B,6]
            R_pred = self.rotFmat.forward(R_pred)                # [B,3,3]

        # Case 2: prediction is a windowed batch [B*win_len,6]
        elif R_pred.ndim == 3 and R_pred.shape[-1] == 6:    # [B,win,6]
            B = R_pred.shape[0]
            R_pred_flat = R_pred.reshape(-1, 6)             # [B*win,6]
            R_pred = self.rotFmat.forward(R_pred_flat)           # [B*win,3,3]

        # Case 3: already a rotation matrix [B,3,3]
        # → do nothing
        elif R_pred.ndim == 3:
            pass

        else:
            raise ValueError(f"Unexpected R_pred shape: {R_pred.shape}")

        # ---------------------------------------------------------
        # Ensure true rotations are proper matrices
        # ---------------------------------------------------------
        R_pred = self.pfm.forward(R_pred)    # normalize orthonormal
        R_true = self.tfm.forward(R_true)

        # ---------------------------------------------------------
        # Compute geodesic rotation error
        # ---------------------------------------------------------
        R_err = R_pred @ R_true.transpose(1, 2)              # R_err = R_pred * R_true^T
        trace = R_err.diagonal(dim1=1, dim2=2).sum(dim=1)    # batch trace
        cos_t = ((trace - 1.0) / 2.0).clamp(-1+eps, 1-eps)    # numerical safety
        theta = torch.acos(cos_t)                            # radians
        deg = theta * 180.0 / torch.pi                       # degrees

        return deg.mean()    # return mean batch rotation error in ° 
    
    def init_params(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, GCNConv):
                # GCNConv stores weights differently, use glorot
                glorot(m.lin.weight)
                if m.lin.bias is not None:
                    zeros(m.lin.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
