import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import os
from CorrNN import weighted_corrnn
from scipy.spatial.transform import Rotation as R


class train_weighted_corr_NN:
    def __init__(self, input_data=None, target_data=None, dataloader=None, weights=None, val_dataloader=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_class = weighted_corrnn

        if dataloader is None:
            assert input_data is not None and target_data is not None, "Need data or a dataloader"
            self.input = input_data
            self.target = target_data
            self.weights = weights if weights is not None else None
            self.dataloader = None
        else:
            self.train_dataloader = dataloader  # already prepared DataLoader
            self.val_dataloader = val_dataloader
            self.input, self.target, self.weights = None, None, None
        
        self.writer = SummaryWriter(log_dir="./runs/corrnn")
    
        self.build_model()
    
    def build_model(self):
        dummy = self.input[0:1, :] if self.input is not None else next(iter(self.train_dataloader))[0][0:1, :]
        self.model = weighted_corrnn(dummy).to(self.device)

    
    def train_loop(self, num_epochs=50, batch_size=64, lr=1e-4, patience=10):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            # ----------------------- TRAIN -----------------------
            self.model.train()
            train_loss = 0.0

            for x, y, H in self.train_dataloader:

                x, y, H = x.to(self.device), y.to(self.device), H.to(self.device)

                q_true = y[:, :4]
                t_true = y[:, 4:]

                optimizer.zero_grad()
                pred = self.model(x, H)

                q_pred = pred[:, :4]
                t_pred = pred[:, 4:]

                #print(f"{q_pred}, {q_true}, t_pred, t_true)")

                loss = self.model.loss(q_pred, q_true, t_pred, t_true)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_dataloader)

            # ----------------------- VALIDATION ------------------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, H in self.val_dataloader:
                    x, y, H = x.to(self.device), y.to(self.device), H.to(self.device)
                    q_true, t_true = y[:, :4], y[:, 4:]
                    pred = self.model(x, H)
                    q_pred, t_pred = pred[:, :4], pred[:, 4:]
                    val_loss += self.model.loss(q_pred, q_true, t_pred, t_true).item()

            val_loss /= len(self.val_dataloader)

            print(f"val loss: {val_loss} | train_loss: {train_loss}")


    def test_loop(self):
        self.model.eval()
        outputs = []

        with torch.no_grad():
            # if dataloader exists, use it
            if hasattr(self, 'dataloader') and self.dataloader is not None:
                for batch in self.dataloader:
                    x = batch[0].to(self.device)
                    theta = self.model(x)   # [batch, 3]
                    outputs.append(theta.cpu())
                outputs = torch.cat(outputs, dim=0)  # [N, 3]
            else:
                # fall back to using raw input_data
                theta = self.model(self.input.to(self.device))
                outputs = theta.cpu()

        # Optional: Plot some stats
        plt.figure()
        plt.plot(torch.mean(torch.abs(outputs), dim=1))
        plt.title("Mean |θ| over time")

        plt.figure()
        plt.hist(outputs.numpy().ravel(), bins=50)

        return outputs

    

    def save_model(self, filepath="corrnn_model.pt"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")



    
    def load_and_test(self, filepath="BASIS_FCN.pt", plot=True):
        print("LOAD AND TEST")

        # Load weights
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {filepath}")

        # Run inference
        with torch.no_grad():
            pred = self.model(self.input.to(self.device)).cpu().numpy()

        # Split prediction
        q_pred = pred[:, :4]
        t_pred = pred[:, 4:]

        # Split ground truth
        q_true = self.target[:, :4]
        t_true = self.target[:, 4:]

        # Convert to Euler (same convention you are using in evaluate)
        euler_pred = R.from_quat(q_pred).as_euler('yzx', degrees=True)
        euler_true = R.from_quat(q_true).as_euler('yzx', degrees=True)

        if plot:
            # ---- Plot Euler ----
            labels = ['θx', 'θy', 'θz']
            for i in range(3):
                plt.figure()
                plt.plot(euler_true[:, i], '--', label=f"True {labels[i]}", alpha=0.6)
                plt.plot(euler_pred[:, i], label=f"Pred {labels[i]}")
                plt.title(f"{labels[i]} (degrees)")
                plt.legend()

            # ---- Plot Translations ----
            t_labels = ['tx', 'ty', 'tz']
            for i in range(3):
                plt.figure()
                plt.plot(t_true[:, i], '--', label=f"True {t_labels[i]}", alpha=0.6)
                plt.plot(t_pred[:, i], label=f"Pred {t_labels[i]}")
                plt.title(f"{t_labels[i]}")
                plt.legend()

            plt.show()

        # Return full pose prediction
        return torch.tensor(pred, dtype=torch.float32)
    

    def evaluate_and_plot(self, use_val=True):
        dl = self.val_dataloader if (use_val and self.val_dataloader is not None) else self.train_dataloader
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y, H in dl:
                x, y, H = x.to(self.device), y.to(self.device), H.to(self.device)
                out = self.model(x, H).cpu().numpy()
                preds.append(out); truths.append(y.cpu().numpy())

        preds = np.vstack(preds); truths = np.vstack(truths)
        q_pred, t_pred = preds[:, :4], preds[:, 4:]
        q_true, t_true = truths[:, :4], truths[:, 4:]

        euler_pred = R.from_quat(q_pred).as_euler('yzx', degrees=True)
        euler_true = R.from_quat(q_true).as_euler('yzx', degrees=True)

        # angles
        for i, lab in enumerate(['θx','θy','θz']):
            plt.figure()
            plt.plot(euler_true[:, i], '--', alpha=0.6, label=f"true {lab}")
            plt.plot(euler_pred[:, i], label=f"pred {lab}")
            plt.legend(); plt.title(f"{lab} (deg)")

        # translations
        for i, lab in enumerate(['tx','ty','tz']):
            plt.figure()
            plt.plot(t_true[:, i], '--', alpha=0.6, label=f"true {lab}")
            plt.plot(t_pred[:, i], label=f"pred {lab}")
            plt.legend(); plt.title(lab)

        plt.show()
        return preds






'''
OLD TEST_LOOP THAT WORKS:


    def train_loop(self, num_epochs=50, batch_size=64, lr=1e-4):
        print("TRAINING")
        self.build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Build dataloader if not provided
        if self.dataloader is None:
            if self.weights is None:
                dataset = TensorDataset(self.input, self.target)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            else:
                dataset = TensorDataset(self.input, self.target, self.weights)
                sampler = WeightedRandomSampler(weights=self.weights.cpu(), num_samples=len(self.weights), replacement=True)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            
            dataloader = self.dataloader

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_q = 0.0
            total_t = 0.0
            for batch in dataloader:
                if len(batch) == 3:
                    h, v_true, w_batch = batch   # weights included
                    w_batch = w_batch.to(self.device)
                else:
                    h, v_true = batch
                    w_batch = None               # no weights used

                h, v_true = h.to(self.device), v_true.to(self.device)

                optimizer.zero_grad()
                pred = self.model(h)

                q_true = v_true[:, :4]
                t_true = v_true[:, 4:]
                
                # your loss returns tuple -> loss, cos_loss
                loss = self.model.loss(pred, q_true, t_true)
                
                # apply weights if given
                if w_batch is not None:
                    loss = (loss * w_batch).mean()
                else:
                    loss = loss.mean() if loss.ndim > 0 else loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * h.size(0)


            mean_loss = total_loss / len(dataloader.dataset)
            mean_q = total_q / len(dataloader.dataset)
            mean_t = total_q / len(dataloader.dataset)

            print(f"Epoch {epoch+1:02d} | Mean loss: {mean_loss:.6f}")

'''