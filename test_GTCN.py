import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import os
from gcntcn import GCNTCN
from scipy.spatial.transform import Rotation as R
import math
from utils.rot6d import RotationTransformer
from utils.represent_data import DataRepresentation



class test_GCTCN:
    def __init__(self, test_dataloaders=None, num_channels=None,
                 t_mean=None, t_std=None, kernel_size=None, gcn_out_dim=7, edge_index_sets=None, use_traj=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_channels = num_channels
        self.t_mean = t_mean
        self.t_std = t_std

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.out_dim = gcn_out_dim
        self.edge_index_sets = edge_index_sets

        self.orientation = 'XYZ'


        self.test_dataloaders = test_dataloaders
        self.input, self.target = None, None

        #self.rotFeul = RotationTransformer('rotation_6d', 'euler_angles', to_convention='YZX')
        #self.quatFeul = RotationTransformer('quaternion', 'euler_angles',to_convention='YZX')
        #self.matFeul = RotationTransformer('matrix', 'euler_angles',to_convention='YZX')

        self.rotFeul = RotationTransformer('rotation_6d', 'euler_angles', to_convention=self.orientation)
        self.quatFeul = RotationTransformer('quaternion', 'euler_angles',to_convention=self.orientation)
        self.matFeul = RotationTransformer('matrix', 'euler_angles',to_convention=self.orientation)


        self.rotFmat = RotationTransformer('rotation_6d', 'matrix')
        self.quatFmat = RotationTransformer('quaternion', 'matrix')
    
    
    def rot_error_deg(self, R_pred, R_true, R_aligned=None, eps=1e-6):

        if R_aligned is None:
            R_pred = R_pred[...,:4]

            # Ensure both are rotation matrices [T,3,3]
            R_pred = self.quatFmat.forward(R_pred)                     # [T,3,3]
            R_true = self.quatFmat.forward(R_true)                     # [T,3,3]

            # Compute relative rotation per timestep
            R_err = R_pred @ R_true.transpose(1, 2)               # [T,3,3]

            # Geodesic SO(3) angle per timestep
            trace = R_err.diagonal(dim1=1, dim2=2).sum(dim=1)     # [T]
            cos_t = ((trace - 1.0) / 2.0).clamp(-1+eps, 1-eps)     # [T]
            theta = torch.acos(cos_t)                             # [T] in radians
            deg = theta * 180.0 / torch.pi                        # [T] in degrees

            return deg.mean()  # scalar mean over timesteps
        
        else:
            R_pred = R_pred[...,:4]

            # Ensure both are rotation matrices [T,3,3]
            R_pred = self.quatFmat.forward(R_pred)                     # [T,3,3]

            # Compute relative rotation per timestep
            R_err = R_pred @ R_aligned.transpose(1, 2)               # [T,3,3]

            # Geodesic SO(3) angle per timestep
            trace = R_err.diagonal(dim1=1, dim2=2).sum(dim=1)     # [T]
            cos_t = ((trace - 1.0) / 2.0).clamp(-1+eps, 1-eps)     # [T]
            theta = torch.acos(cos_t)                             # [T] in radians
            deg = theta * 180.0 / torch.pi                        # [T] in degrees

            return deg.mean()  # scalar mean over timesteps                 # [T,3,3]

    
    def compute_global_frame_bias(self, mat_pred, mat_true, eps=1e-6):
        #R_pred = R_pred[...,:4]
        #mat_pred = self.quatFmat.forward(R_pred)                     # [T,3,3]
        #mat_true = self.quatFmat.forward(R_true)                     # [T,3,3]

        # Step 1: Accumulate cross-covariance (Wahba / Kabsch matrix)
        # P = sum_t (R_trueᵀ(t) @ R_pred(t))
        P = (mat_true.transpose(1,2) @ mat_pred).sum(dim=0)   # [3,3]

        # Step 2: SVD
        U, S, Vh = torch.linalg.svd(P)

        # Step 3: Construct determinant-correcting diagonal matrix
        D = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(U @ Vh))],
                                    device=mat_pred.device, dtype=mat_pred.dtype))

        # Step 4: Compute optimal rotation
        R_bias = U @ D @ Vh

        R_aligned = R_bias @ mat_pred

        return R_aligned  # [3,3]


    def rot_error_deg_from_mats(self, mat_pred: torch.Tensor, mat_true: torch.Tensor, eps=1e-6):
        """
        mat_pred, mat_true: [T,3,3] rotation matrices (orthonormalized already).
        Returns numpy array of per-timestep errors in degrees, shape [T].
        """
        R_err = mat_pred @ mat_true.transpose(1, 2)                 # [T,3,3]
        trace = R_err.diagonal(dim1=1, dim2=2).sum(dim=1)           # [T]
        cos_t = ((trace - 1.0) / 2.0).clamp(-1+eps, 1-eps)
        theta = torch.acos(cos_t)                                    # radians
        deg = (theta * 180.0 / torch.pi).detach().cpu().numpy()      # [T]
        return deg
    
    def quat_normalize(self,q: torch.Tensor, eps=1e-8):
        return q / (q.norm(dim=-1, keepdim=True) + eps)
    

    def _summarize_deg(self, deg_np: np.ndarray):
        return {
            "count": int(deg_np.shape[0]),
            "mean": float(np.mean(deg_np)),
            "std":  float(np.std(deg_np)),
            "median": float(np.median(deg_np)),
            "p75": float(np.percentile(deg_np, 75)),
            "p90": float(np.percentile(deg_np, 90)),
            "p95": float(np.percentile(deg_np, 95)),
            "p99": float(np.percentile(deg_np, 99)),
            "max": float(np.max(deg_np)),
        }

    def build_model(self):
        """
        Build the same model architecture used in training.
        Must match layer sizes, num_channels, kernel_size, and output dim (6).
        """
        # Import here to avoid circular imports

        sample_dl = self.test_dataloaders[0]
        x_sample, _, _ = next(iter(sample_dl))   # [B, C, T]
        x_sample = x_sample.to(self.device)

        self.model = GCNTCN(
            input=x_sample,
            edge_index_sets=self.edge_index_sets,
            gcn_hidden_dim=32,
            gcn_out_dim=7,
            num_channels = self.num_channels,
            kernel_size=self.kernel_size
        ).to(self.device)

    def smooth_euler_deg(self, r):
        r = r.copy()

        for i in range(1, len(r)):
            diff = r[i] - r[i-1]

            # Component-wise continuity correction
            r[i][diff > 180]  -= 360
            r[i][diff < -180] += 360

        return r


    def evaluate_trial(self, x: torch.Tensor, y: torch.Tensor, plot: bool = True):
        """
        x: [T,...] or [B,T,...] (we'll flatten batch if present)
        y: [T,7] or [B,T,7]  (ground-truth: quat(4, xyzw) + trans(3))
        Returns: dict with metrics, and (q_pred, t_pred, q_true, t_true) as torch Tensors on CPU.
        """
        self.model.eval()

        # Ensure [T, ...]
        if x.dim() >= 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if y.dim() >= 3 and y.shape[0] == 1:
            y = y.squeeze(0)

        with torch.no_grad():
            pred = self.model(x.to(self.device)).cpu()   # [T,7]
        
        print(pred.shape)
        print(type(pred))

        q_pred = pred[:, :4]
        t_pred = pred[:, 4:]
        q_pred = self.quat_normalize(q_pred)

        q_true = y[:, :4].cpu()
        t_true = y[:, 4:].cpu()


        # Normalize quats (safety)

        # Hemisphere fix for stable raw metric (optional but helps)
        #dots = (q_pred * q_true).sum(dim=-1, keepdim=True)
        #q_pred = q_pred * torch.sign(dots)

        # Rotation matrices
        mat_pred = self.rotFmat.forward(q_pred)        # [N, 6]
        mat_true = self.quatFmat.forward(q_true[:, :4])      # [N, 7]

        # RAW rotation error distribution
        deg_raw = self.rot_error_deg_from_mats(mat_pred, mat_true)
        summary_raw = self._summarize_deg(deg_raw)

        # Aligned rotation error (R2)
        R_bias = self.compute_global_frame_bias(mat_pred, mat_true)       # [3,3]
        mat_pred_aligned = (R_bias @ mat_pred)                       # broadcast over T
        deg_aligned = self.rot_error_deg_from_mats(mat_pred_aligned, mat_true)
        summary_aligned = self._summarize_deg(deg_aligned)

        # Translation denorm (assumes self.t_mean/std broadcastable [1,3] or [3])
        t_pred_den = t_pred * torch.as_tensor(self.t_std).float() + torch.as_tensor(self.t_mean).float()
        t_true_den = t_true * torch.as_tensor(self.t_std).float() + torch.as_tensor(self.t_mean).float()

        # Translation metrics
        err_trans = (t_true_den - t_pred_den).numpy()                # [T,3]
        trans_MAE_axis = np.mean(np.abs(err_trans), axis=0)          # [3]
        trans_axis_std  = np.std(err_trans, axis=0)                  # [3]
        trans_mag = np.linalg.norm(err_trans, axis=1)                # [T]
        trans_mean_mag = float(np.mean(trans_mag))
        trans_std_mag  = float(np.std(trans_mag))

        r_pred = self.quatFeul.forward(q_pred)
        r_true = self.quatFeul.forward(q_true)

        #r_pred = (np.unwrap(r_pred) + np.pi) % (2*np.pi) - np.pi
        #r_true = (np.unwrap(r_true) + np.pi) % (2*np.pi) - np.pi

        r_pred = np.rad2deg(r_pred)
        r_true = np.rad2deg(r_true)

        r_pred = self.smooth_euler_deg(r_pred.numpy())
        r_true = self.smooth_euler_deg(r_true.numpy())

        err_euler = (r_true - r_pred)               # [T,3]
        euler_MAE_axis = np.mean(np.abs(err_euler), axis=0)          # [3]
        euler_axis_std  = np.std(err_euler, axis=0)  
        euler_mag = np.linalg.norm(err_euler, axis=1)                 # [3]
        euler_mean_mag = float(np.mean(euler_mag))
        euler_std_mag  = float(np.std(euler_mag))
        
        '''
                # err = [T, 3] → (x, y, z)
        x = err_euler[:, 0]
        y = err_euler[:, 1]
        z = err_euler[:, 2]

        # magnitude (rotation error norm)
        mag = np.sqrt(x*x + y*y + z*z)      # [T]

        # spherical coordinates
        azimuth = np.arctan2(y, x)          # φ in radians
        elevation = np.arctan2(z, np.sqrt(x*x + y*y))   # θ in radians

        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)'''



        # Print nice summary (keep prints as requested)
        print("Rotation (deg) RAW  :", summary_raw)
        print("Rotation (deg) ALIGN:", summary_aligned)
        print(f"Translation MAE per axis (mm): {trans_MAE_axis} | mean|err|={trans_mean_mag:.3f}  std|err|={trans_std_mag:.3f} | axis_std={trans_axis_std}")
        print(f"Rotation MAE per axis (mm): {euler_MAE_axis} | mean|err|={euler_mean_mag:.3f}  std|err|={euler_std_mag:.3f} | axis_std={euler_axis_std}")

        # Plots (P3: rotations + translations)
        if plot:
            # Euler (degrees) for visualization
            #euler_pred = R.from_quat(q_pred.numpy()).as_euler('self.orientation', degrees=True)
            #euler_true = R.from_quat(q_true.numpy()).as_euler(self.orientation, degrees=True)
            # Optional unwrap (helps visualization)
            #r_pred = np.unwrap(r_pred, axis=0, period=360) 
            #r_true = np.unwrap(r_true, axis=0, period=360)

            #euler_pred[:, 2] = euler_pred[:, 2] * -1

            print(f"r_pred z max {np.max(r_pred[:, 2])}")
            
            labels = ['θx','θy','θz']
            colors = ['r', 'g', 'b']

            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        
            dt = 0.05  # 50 ms per sample
            num_samples = r_true.shape[0]
            time = np.arange(num_samples) * dt
            
        for i in range(3):
                axs[i].plot(time[9000:16000], r_true[9000:16000, i], linestyle='--', color=colors[i], alpha=0.7, label=f"True {labels[i]}")
                axs[i].plot(time[9000:16000], r_pred[9000:16000, i], linestyle='-',  color=colors[i], alpha=0.9, label=f"Pred {labels[i]}")
                
                axs[i].set_ylabel(f"{labels[i]} (deg)")
                axs[i].legend()
                axs[i].grid(True)

        axs[-1].set_xlabel("Time (s)")
        plt.suptitle("Euler Angles (deg)")
        plt.tight_layout()
        plt.show()
            
            
        t_labels = ['tx','ty','tz']
        colors = ['r','g','b']

        t_pred_den_np = t_pred_den.numpy()
        t_true_den_np = t_true_den.numpy()

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        for i in range(3):
            axs[i].plot(time[9000:16000],t_true_den_np[9000:16000, i],  '--', color=colors[i], alpha=0.6, label=f"True {t_labels[i]}")
            axs[i].plot(time[9000:16000],t_pred_den_np[9000:16000, i],  '-', color=colors[i], alpha=0.9, label=f"Pred {t_labels[i]}")
            axs[i].set_ylabel(f"{t_labels[i]} (mm)")
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel("time(s)")
        fig.suptitle("Translation Trajectories (mm)")
        plt.tight_layout()
        plt.show()
            
        data = torch.load(f"datasets/Val/0.pt")
        sc = data["sc"]
        capi = data["capi"]
        Dv = DataRepresentation(sc_array=sc, capi_array=capi)
        pressure = Dv.relative_pressure_mag

        plt.figure()
        plt.title("Pressure Signals")
        plt.plot(pressure[9000:16000, :])
        plt.ylabel("Voltage")
        plt.xlabel("Time (s)")

        plt.show()


        # Pack metrics (P3: return metrics too)
        metrics = {
            "rot_deg_raw": summary_raw,
            "rot_deg_aligned": summary_aligned,

            "euler_mae_axis_mm": euler_MAE_axis.tolist(),
            "euler_axis_std_mm": euler_axis_std.tolist(),
            "euler_mean_mag_mm": euler_mean_mag,
            "euler_std_mag_mm": euler_std_mag,

            "trans_mae_axis_mm": trans_MAE_axis.tolist(),
            "trans_axis_std_mm": trans_axis_std.tolist(),
            "trans_mean_mag_mm": trans_mean_mag,
            "trans_std_mag_mm": trans_std_mag,
        }

        return (q_pred, t_pred, q_true, t_true), metrics


    # ---------- Main entry: iterate trials (A1) ----------

    def load_and_test(self, filepath="", plot=True):
        """
        Expects: self.test_dataloaders = [DataLoader(ds1), DataLoader(ds2), ...]
        Returns (RET3): (predictions_per_trial, metrics_per_trial)
        - predictions_per_trial: list of tuples (q_pred, t_pred, q_true, t_true) per trial (torch CPU tensors)
        - metrics_per_trial: list of dicts per trial
        """

        # Build model first
        self.build_model()
        self.model.to(self.device)

        # Load weights
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {filepath}")

        assert hasattr(self, "test_dataloaders") and isinstance(self.test_dataloaders, list) \
            and len(self.test_dataloaders) > 0, "Provide a list of test dataloaders (one per trial)."

        preds_per_trial = []
        metrics_per_trial = []

        with torch.no_grad():
            for trial_idx, dl in enumerate(self.test_dataloaders):
                # Collect the whole trial (preserve continuity)
                xs, ys = [], []
                for x, y, _ in dl:
                    xs.append(x)  # keep in the same order
                    ys.append(y)
                x_trial = torch.cat(xs, dim=0)  # [T,...]
                y_trial = torch.cat(ys, dim=0)  # [T,7]

                print(f"[Test] Trial {trial_idx+1}/{len(self.test_dataloaders)}: evaluating...")
                (q_pred, t_pred, q_true, t_true), metrics = self.evaluate_trial(x_trial, y_trial, plot=plot)

                preds_per_trial.append((q_pred, t_pred, q_true, t_true))
                metrics_per_trial.append(metrics)

        return preds_per_trial, metrics_per_trial



   