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



class train_GCTCN:
    def __init__(self, input_data=None, target_data=None, dataloader=None, edge_index_sets=None, val_dataloader=None, 
                 test_dataloader=None, num_channels=None,
                 t_mean=None, t_std=None, kernel_size=None, use_traj=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_index_sets = edge_index_sets
        self.num_channels = num_channels
        self.t_mean = t_mean
        self.t_std = t_std
        self.kernel_size =kernel_size

        self.out_layer_dim = 6

        # Two modes: (1) dataloader mode, (2) raw tensor mode
        if dataloader is None and test_dataloader is None:
            assert input_data is not None and target_data is not None, "Need data or a dataloader"
            # Ensure tensors on CPU initially; move to device in forward calls
            self.input = torch.as_tensor(input_data).contiguous()
            self.target = torch.as_tensor(target_data).contiguous()
            self.train_dataloader = None
            self.val_dataloader = None
        else:
            self.train_dataloader = dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader
            self.input, self.target = None, None

        self.rotFeul = RotationTransformer('rotation_6d', 'euler_angles', to_convention='YZX')
        self.quatFeul = RotationTransformer('quaternion', 'euler_angles',to_convention='YZX')
        self.matFeul = RotationTransformer('matrix', 'euler_angles',to_convention='YZX')


        self.rotFmat = RotationTransformer('rotation_6d', 'matrix')
        self.quatFmat = RotationTransformer('quaternion', 'matrix')

        self.build_model()

    def _get_dummy_from_data(self):
        """
        Returns a 3D tensor dummy of shape [1, 11, T] on CPU, inferred from either
        the first batch of the train_dataloader or from self.input in raw-tensor mode.
        """
        if self.train_dataloader is not None:
            xb, yb, tb = next(iter(self.train_dataloader))  # xb: [B, 11, T]
            assert xb.dim() == 3, f"Expect [B, 11, T], got {tuple(xb.shape)}"
            dummy = xb[:1].contiguous()                 # [1, 11, T]
        
        elif self.test_dataloader is not None:
            xb, yb, tb = next(iter(self.test_dataloader))  # xb: [B, 11, T]
            assert xb.dim() == 3, f"Expect [B, 11, T], got {tuple(xb.shape)}"
            dummy = xb[:1].contiguous()                 # [1, 11, T]
        
        else:
            x = self.input
            # Accept either [B, 11, T] or [11, T]
            if x.dim() == 3:
                dummy = x[:1].contiguous()              # [1, 11, T]
            elif x.dim() == 2:
                # Add batch dimension
                dummy = x.unsqueeze(0).contiguous()     # [1, 11, T]
                assert dummy.shape[1] == 11, f"Expect channels=11, got {dummy.shape}"
            else:
                raise ValueError(f"Unsupported input shape {tuple(x.shape)}; need [B,11,T] or [11,T].")
        return dummy

    def build_model(self):
        dummy = self._get_dummy_from_data()  # CPU tensor [1, 11, T]
        # Use your constructor signature and dims you chose earlier
        self.model = GCNTCN(
            input=dummy,                      # shape only; used to set internal dims
            edge_index_sets=self.edge_index_sets,
            gcn_hidden_dim=32,
            gcn_out_dim=7,
            num_channels = self.num_channels,
            kernel_size=self.kernel_size,
            
        ).to(self.device)

    def save_model(self, filepath="corrnn_model.pt"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    
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

    
    def compute_global_frame_bias(self, R_pred, R_true, eps=1e-6):
        R_pred = R_pred[...,:4] # NOTE: NEED TO NORMALIZE
        
        mat_pred = self.quatFmat.forward(R_pred)                     # [T,3,3]
        mat_true = self.quatFmat.forward(R_true)                     # [T,3,3]

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



    def train_loop(self, num_epochs=50, lr=1e-4, patience=10):
        assert self.train_dataloader is not None, "train_loop requires a dataloader"

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            # ----------------------- TRAIN -----------------------
            self.model.train()
            train_loss_sum, train_batches = 0.0, 0
            rot_loss = 0.0, 0.0
            l1_lambda = 1e-5

            for x, y, traj in self.train_dataloader:
                x = x.to(self.device)          # [B, 11, T]
                y = y.to(self.device)          # [B, 7]
                traj = traj.to(self.device)

                q_true = y[:, :4]   # True Quaternion
                t_true = y[: , 4:]  # True translation
                
                optimizer.zero_grad()
                if torch.isnan(x).any():
                    raise RuntimeError("Input exploded into NaN before model forward")
               
                pred = self.model(x)           # [B, 4] Quaternion

                #q_pred = pred[:, :4]    # Predicted Quaternion

                if torch.isnan(pred).any():
                    raise RuntimeError("Model output NaN before loss")

                # Quaternion -> RotationMatrix Loss
                loss = self.model.loss(s_pred=pred, q_true=q_true,trajectory=t_true)
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())

                loss = loss + l1_norm*l1_lambda

                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)

            # ----------------------- VALIDATION ------------------
            val_loss = float("nan")
            if self.val_dataloader is not None:
                self.model.eval()
                val_loss_sum, val_batches = 0.0, 0
                val_q_loss, val_t_loss = 0.0, 0.0
                with torch.no_grad():
                    for x, y, _ in self.val_dataloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        val_q_true = y[:, :4]   # True Quaternion
                        val_t_true = y[:, 4:]   # True Quaternion

                        val_pred = self.model(x)

                        loss_val = self.model.loss(val_pred, val_q_true, trajectory=val_t_true)
                        val_loss_sum += float(loss_val.item())
                        val_batches += 1
                
            
                R_aligned = self.compute_global_frame_bias(val_pred, val_q_true)
                
                val_loss = val_loss_sum / max(val_batches, 1)
                scheduler.step(val_loss)


                print(f"Train: {self.rot_error_deg(pred, q_true)}")
                print(f"Validation: {self.rot_error_deg(val_pred, val_q_true,R_aligned=R_aligned)}")
            
            # ---- EARLY STOPPING ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break



    def evaluate_and_plot(self, use_val=True, traj=False):
        dl = self.val_dataloader if (use_val and self.val_dataloader is not None) else self.train_dataloader
        assert dl is not None, "No dataloader available for evaluation."
        self.model.eval()

        preds, truths, trajs = [], [], []

        with torch.no_grad():
            for x, y, traj_t in dl:
                x = x.to(self.device)
                y = y.to(self.device)
                traj_t = traj_t.to(self.device)

                out = self.model(x)              # [B, 6] or [B, 3] depending on model
                preds.append(out.cpu())
                truths.append(y.cpu())
                trajs.append(traj_t.cpu())

        preds  = torch.vstack(preds)            # [N, 6] or [N, 3]
        truths = torch.vstack(truths)           # [N, 6] or [N, 3]
        trajs  = torch.vstack(trajs)            # [N, 3]


        # ---- convert to rotation matrices [N, 3, 3] ----
        mat_pred = self.rotFmat.forward(preds[:, :4])        # [N, 6]
        mat_true = self.quatFmat.forward(truths[:, :4])      # [N, 7]

        # ---- Euler conversion ----
        euler_pred = self.matFeul.forward(mat_pred).numpy()
        euler_true = self.matFeul.forward(mat_true).numpy()


        deg = self.rot_error_deg_from_mats(mat_pred, mat_true)

        summary = {
            "count": len(deg),
            "mean": float(np.mean(deg)),
            "std": float(np.std(deg)),
            "median": float(np.median(deg)),
            "p75": float(np.percentile(deg, 75)),
            "p90": float(np.percentile(deg, 90)),
            "p95": float(np.percentile(deg, 95)),
            "p99": float(np.percentile(deg, 99)),
            "max": float(np.max(deg)),
        }
        print(summary)


        err = np.abs(euler_true - euler_pred)
        euler_MAE = np.mean(err,axis=0)
        mean_err = np.mean(euler_MAE)
        std_err  = np.std(euler_MAE)
        axis_std = np.std(err, axis=0)
        print(F"EULER MAE: {euler_MAE} mean: {mean_err} std: {std_err}\n std per axis: {axis_std}")


        torch.save({"euler_pred": euler_pred, "euler_true": euler_true}, 
                    f"model_results/gctn_euler.pt")

        # ---- plot Euler angles ----
        for i, lab in enumerate(['θx','θy','θz']):
            plt.figure()
            plt.plot(euler_true[:, i], '--', alpha=0.6, label=f"true {lab}")
            plt.plot(euler_pred[:, i], label=f"pred {lab}")
            plt.legend(); plt.title(f"{lab} (deg)")
        plt.show()


        t_pred = preds[:, 4:].numpy()
        t_true = truths[:, 4:].numpy()

        t_pred_denorm = t_pred * self.t_std + self.t_mean
        t_true_denorm = t_true * self.t_std + self.t_mean


        err = np.abs(t_true_denorm - t_pred_denorm)
        trans_MAE = np.mean(err,axis=0)
        mean_err = np.mean(trans_MAE)
        std_err  = np.std(trans_MAE)
        axis_std = np.std(err, axis=0)
        print(F"TRANSLATION MAE: {trans_MAE} mean: {mean_err} std: {std_err}\n std per axis: {axis_std}")

        for i, lab in enumerate(['tx','ty','tz']):
            plt.figure()
            plt.plot(t_true[:, i], '--', alpha=0.6, label=f"true {lab}")
            plt.plot(t_pred[:, i], label=f"pred {lab}")
            plt.legend(); plt.title(lab)
        plt.show()



    def train_loop_traj(self, num_epochs=50, lr=5e-5, patience=10):
        assert self.train_dataloader is not None, "train_loop requires a dataloader"
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            # ----------------------- TRAIN -----------------------
            self.model.train()
            train_loss_sum, train_batches = 0.0, 0
            rot_loss = 0.0, 0.0
            l1_lambda = 1e-4

            unit_vector = torch.Tensor([-44.32, 40.45, 50])

            for x, y, traj in self.train_dataloader:
                x = x.to(self.device)          # [B, T, 11]
                y = y.to(self.device)          # [B, 7]
                traj = traj.to(self.device)     # [B, T, 3]
                unit_vector = unit_vector.to(self.device)

                q_true = y[:, :, :4]   # True Quaternion
                
                optimizer.zero_grad()
                if torch.isnan(x).any():
                    raise RuntimeError("Input exploded into NaN before model forward")
                pred = self.model(x)           # [B, 6] 6D-rotation matrix

                d6_pred = pred    # Predicted 6D-rot vecot

                if torch.isnan(pred).any():
                    raise RuntimeError("Model output NaN before loss")

                # Quaternion -> 6D -> RotationMatrix Loss
                loss = self.model.loss_traj(s_pred=d6_pred, r_true=q_true, trajectory=traj, unit_vector=unit_vector)

                l1_norm = sum(p.abs().sum() for p in self.model.parameters())

                loss = loss + l1_norm*l1_lambda

                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)

            # ----------------------- VALIDATION ------------------
            val_loss = float("nan")
            if self.val_dataloader is not None:
                self.model.eval()
                val_loss_sum, val_batches = 0.0, 0
                val_q_loss, val_t_loss = 0.0, 0.0
                with torch.no_grad():
                    for x, y, traj in self.val_dataloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        traj = traj.to(self.device)

                        val_q_true = y[:, :, :4]  # Quaternion
                        val_pred = self.model(x)

                        l_val = self.model.loss_traj(val_pred, val_q_true, traj, unit_vector)
                        val_loss_sum += float(l_val.item())
                        val_batches += 1
                
                val_loss = val_loss_sum / max(val_batches, 1)
                scheduler.step(val_loss)

                print(f"Train: {train_loss}")
                print(f"Validation: {val_loss}")
            
            # ---- EARLY STOPPING ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self.model.load_state_dict(best_state)
                    break