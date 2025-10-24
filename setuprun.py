from utils.mydataloader2 import construct_data
from utils.MatchIOFiles import matching_io
from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.represent_data import DataRepresentation
from run_CorrNN import train_weighted_corr_NN

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset, ConcatDataset

class setup_and_run:
    
    def __init__(self, trainpath=None, testpath=None, train=False):
        self.trainpath = trainpath
        self.testpath = testpath
        self.train = train
    
    def setup_data(self, val_trials=1, batch_size=64):
            
        if self.train:
            
            io_pair = np.array(matching_io(self.trainpath + "/sc", self.trainpath + "/capi"))

            trial_datasets = []
            val_trial_datasets = []

            # --- load each trial, compute its own H, and pack (x,y,Hrep) ---
            for i in range(len(io_pair)):
                CAPIreader = ReadCapi2(io_pair[i, 0])
                SCreader   = ReadSC2(io_pair[i, 1])

                Synced = AlignData(CAPIreader, SCreader)
                Synced.check_timezone()
                Synced.interpret2(delete_missing=True)

                x_np = Synced.adjusted_sc_clean.numpy()    # [N, 11]
                y_np = Synced.adjusted_capi_clean.numpy()  # [N, 7]

                V = np.array([1,1,0])
                D  = DataRepresentation(sc_array=x_np, capi_array=y_np, V=V)

                X = torch.tensor(D.input_data,  dtype=torch.float32)           # [N, 11]
                Y = torch.tensor(D.target_data, dtype=torch.float32)           # [N, 7]
                H = torch.tensor(D.H,           dtype=torch.float32)           # [sensors, k]
                Hrep = H.unsqueeze(0).repeat(X.size(0), 1, 1)                  # [N, sensors, k]

                ds = TensorDataset(X, Y, Hrep)
                trial_datasets.append(ds)

            # ---- split by trial (last `val_trials` are validation) ----
            train_trials = trial_datasets[:-val_trials] if val_trials > 0 else trial_datasets
            val_trials_ds = trial_datasets[-val_trials:] if val_trials > 0 else []

            train_ds = ConcatDataset(train_trials)
            self.train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            if val_trials_ds:
                val_ds = ConcatDataset(val_trials_ds)
                self.val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            else:
                self.val_dataloader = None

            print(f"✅ Trials total: {len(trial_datasets)} | Train trials: {len(train_trials)} | Val trials: {len(val_trials_ds)}")


        
        else:
            print("TESTING")

            io_pair = np.array(matching_io(self.testpath + "/sc", self.testpath + "/capi"))

            for i in range(len(io_pair)):
                CAPIreader = ReadCapi2(io_pair[i, 0])  
                SCreader   = ReadSC2(io_pair[i, 1])  

                # Sync SC + CAPI
                Synced = AlignData(CAPIreader, SCreader)
                Synced.check_timezone()
                Synced.interpret2(delete_missing=True)

                # Synced.adjusted_sc_clean -> torch [N,11]
                # Synced.adjusted_capi_clean -> torch [N,7] (quat+trans) after your Realign code

                self.input_data  = Synced.adjusted_sc_clean.numpy()       # [N,11]
                self.target_data = Synced.adjusted_capi_clean.numpy()     # [N,7]

                print("input:",  self.input_data.shape)
                print("target:", self.target_data.shape)

                # ---- run your DataRepresentation just like before ----
                self.V = np.array([1,1,0])    
                DataObject = DataRepresentation(sc_array=self.input_data, capi_array=self.target_data, V=self.V)
                self.H = DataObject.H

                # overwrite with aligned data
                input_tensor  = torch.tensor(DataObject.input_data,  dtype=torch.float32)
                target_tensor = torch.tensor(DataObject.target_data, dtype=torch.float32)

                trainer = train_weighted_corr_NN(
                input_data=input_tensor,
                target_data=target_tensor,
                unit_vec=self.V,
                dataloader=None,     # TRAIN DL
                val_dataloader=None,   # VAL DL
                )

                trainer.load_and_test("finaldraft1.pt")

    
    def run(self, filepath=None):

        trainer = train_weighted_corr_NN(
            dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader
        )

        if filepath is None:
            trainer.train_loop(num_epochs=15, batch_size=64, lr=1e-4)
            trainer.evaluate_and_plot()
            trainer.save_model("finaldraft_fixedH.pt")
        else:
            trainer.load_and_test(filepath)