from utils.mydataloader2 import construct_data
from utils.MatchIOFiles import matching_io
from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.represent_data import DataRepresentation
from myutils import get_feature_map, get_tactile_graph_struc, build_loc_net, get_fc_graph_struc
from utils.myTimeDataset import TimeDataset

from run_GCTCN import train_GCTCN
from test_GTCN import test_GCTCN

import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset, ConcatDataset

class setup_and_run:
    
    def __init__(self, trainpath=None, testpath=None, train=True, lr=None, 
                 slide_win=128,slide_stride=1, epochnum = 15, graph_name='gg',num_channels=None, kernel_size=None, traj=True,pathtest=None):
        self.trainpath = trainpath
        self.testpath = testpath
        self.train = train
        self.slide_win= slide_win
        self.slide_stride= slide_stride
        self.graph_name = graph_name
        self.epochnum = epochnum
        self.num_channels = num_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generate_edgeIdx()
        self.kernel_size = kernel_size
        self.lr = lr
        self.traj= traj
        self.pathtest = pathtest

        self.num_files = len(glob.glob("datasets/Train/*.pt"))  # or *.npy, *.pt, etc.
        print("Number of files:", self.num_files)
    

    def setup_data(self, val_trials=1, batch_size=64):
            
        if self.train:
            trial_datasets = []
            val_trial_datasets = []
            transform_stats = []

            # --- load each trial, and save pose mean and std) ---
            for i in range(self.num_files):
            #for i in range(2):
                
                if i==0 or i==self.num_files+1:    # NOTE: Skip datasets
                    continue
                
                data = torch.load(f"datasets/Train/{i}.pt")

                sc = data["sc"]
                capi = data["capi"]

                D = DataRepresentation(sc_array=sc, capi_array=capi)
                transform_stats.append(D.all_data[:, 11+4:11+7])  # this is [:, 4:7] of the capi part
            

            # compute global z-score
            t_all = np.vstack(transform_stats)                # [sum Ni, 3]
            self.t_mean = t_all.mean(axis=0, keepdims=True)   # [1, 3]
            self.t_std  = t_all.std(axis=0, keepdims=True)     # [1, 3]

            print(f"t mean: {self.t_mean}")
            print(f"t std: {self.t_std}")


            # --- load each trial for TimeDataset ---
            for i in range(self.num_files):
            #for i in range(2):
                
                if i==0 or i==self.num_files+1:    # NOTE: Skip datasets
                    continue
                
                data = torch.load(f"datasets/Train/{i}.pt")
                sc = data["sc"]
                capi = data["capi"]

                D = DataRepresentation(sc_array=sc, capi_array=capi)

                dataset = TimeDataset(D.all_data,slide_win=self.slide_win,slide_stride=self.slide_stride, 
                                      t_mean=self.t_mean, t_std=self.t_std,traj=self.traj)  # [N, 11], [N, 7], [N, 3]
                
                trial_datasets.append(dataset)
                        
            # val dataset path:
            data = torch.load(f"datasets/Val/0.pt")
            sc = data["sc"]
            capi = data["capi"]
            Dv = DataRepresentation(sc_array=sc, capi_array=capi)
            val_trials_ds = [TimeDataset(Dv.all_data,slide_win=self.slide_win,slide_stride=self.slide_stride, 
                                         t_mean=self.t_mean, t_std=self.t_std, traj=self.traj)]
            

            # ---- split by trial (last `val_trials` are validation) ----
            train_trials = trial_datasets[:-val_trials] if val_trials > 0 else trial_datasets
            #val_trials_ds = trial_datasets[-val_trials:] if val_trials > 0 else []

            train_ds = ConcatDataset(train_trials)
            self.train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            if val_trials_ds:
                val_ds = ConcatDataset(val_trials_ds)
                self.val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            else:
                self.val_dataloader = None

            print(f"Trials total: {len(trial_datasets)} | Train trials: {len(train_trials)} | Val trials: {len(val_trials_ds)}")


        
        else:
            print("TESTING")

            # Hardcoded (correct) training statistics
            t_mean = np.array([-133.75, -202.68, -94.29])
            t_std  = np.array([5.11, 9.65, 21.52])

            self.test_dataloaders = []

            for i in range(self.num_files):

                # Skip datasets by rule
                if i == 0 or i == self.num_files + 1:
                    continue

                data = torch.load(f"TacGat/datasets/Train/{i}.pt")
                sc = data["sc"]
                capi = data["capi"]

                D = DataRepresentation(sc_array=sc, capi_array=capi)

                ds = TimeDataset(
                    D.all_data,
                    slide_win=self.slide_win,
                    slide_stride=self.slide_stride,
                    t_mean=t_mean,
                    t_std=t_std,
                    traj=self.traj,
                    mode='Test'
                )

                # A1: one DataLoader per trial
                dl = DataLoader(ds, shuffle=False, batch_size=1)
                self.test_dataloaders.append(dl)

            print(f"[TEST] Loaded {len(self.test_dataloaders)} trials.")
 

    
    def run(self, filepath=None, traj=False):


        if self.train:
                    
                trainer = train_GCTCN(
                    dataloader=self.train_dataloader,
                    val_dataloader=self.val_dataloader,
                    edge_index_sets = self.edge_index_sets,
                    num_channels=self.num_channels,
                    t_mean=self.t_mean,
                    t_std=self.t_std,
                    kernel_size=self.kernel_size
                )

                if filepath is None: 
                
                    if not traj:
                        trainer.train_loop(num_epochs=self.epochnum, lr=self.lr)
                        trainer.evaluate_and_plot(traj=traj)
                        trainer.save_model(self.pathtest)

                    else: 
                        print("using trajpred")
                        trainer.train_loop_traj(num_epochs=self.epochnum, lr=self.lr)
                        trainer.evaluate_and_plot(traj=traj)
                        trainer.save_model(self.pathtest)
                    
                else:
                    trainer.load_and_test(filepath)

        else:
            t_mean = np.array([-133.75, -202.68, -94.29])
            t_std = np.array([5.11, 9.65, 21.52])
            trainer = test_GCTCN(
                test_dataloaders=self.test_dataloaders,
                edge_index_sets = self.edge_index_sets,
                t_mean=t_mean,
                t_std=t_std,
                kernel_size=self.kernel_size,
                num_channels=self.num_channels
                )
            # sequence inference over the sliding windows
            
            preds = trainer.load_and_test(self.pathtest, plot=True) 
            
    

    def generate_edgeIdx(self):
        """Initialize the model based on graph structure."""
        feature_map = get_feature_map()
        self.feature_map = feature_map
        
        edge_index_sets = []
        
        fc_struc = get_fc_graph_struc(self.feature_map)
        tactile_struc = get_tactile_graph_struc(self.feature_map)

        fc_edge_index = build_loc_net(fc_struc, self.feature_map)
        tactile_edge_index = build_loc_net(tactile_struc, self.feature_map)

        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long, device=self.device)
        tactile_edge_index = torch.tensor(tactile_edge_index, dtype=torch.long, device=self.device)


        if self.graph_name == 'g':
            edge_index_sets.append(fc_edge_index)
        elif self.graph_name == 'g+':
            edge_index_sets.append(tactile_edge_index)
        else:
            edge_index_sets.extend([fc_edge_index, tactile_edge_index])
        
        self.edge_index_sets = edge_index_sets
