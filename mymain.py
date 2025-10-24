import torch
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, RandomSampler

import argparse
import numpy as np
import random
from datetime import datetime
from pathlib import Path
import os

from utils.mydataloader2 import construct_data
from utils.MatchIOFiles import matching_io
from utils.ReadCapi import ReadCapi2
from utils.ReadSC import ReadSC2
from utils.Align import AlignData
from utils.represent_data import DataRepresentation

from models.myTactileGAT import TactileGAT
from run_CorrNN import train_corr_NN, train_weighted_corr_NN

from mytrain import train
from mytest import test
from utils.myTimeDataset import TimeDataset
from myutils import get_feature_map, get_tactile_graph_struc, build_loc_net, get_fc_graph_struc
from utils.device import set_device, get_device

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Main:

    def __init__(self, config):
        setup_seed(config['seed'])
        self.datestr = None
        self.config = config
        set_device(config['device'])
        self.device = get_device()
        
        # Need to create list of matching input and target files by reading from directories
        self.path = "/home/jayantdubey/Desktop/TrainData"
        
        self.setup_data()
        print("Setup Data Complete")

        self.setup_model()
        print("Setup Model Complete")
    

    def compute_norm_stats(self,raw_datasets):
        # raw_datasets: list of full raw_data arrays [C, T]
        all_inputs = np.concatenate([d[:11, :] for d in raw_datasets], axis=1)
        all_trans  = np.concatenate([d[15:18, :] for d in raw_datasets], axis=1)  # if 11:14 are quats, 14:17 transl

        # Per-sensor mean/std
        input_mean = all_inputs.mean(axis=1, keepdims=True)
        input_std  = all_inputs.std(axis=1, keepdims=True) + 1e-8

        trans_mean = all_trans.mean(axis=1, keepdims=True)
        trans_std  = all_trans.std(axis=1, keepdims=True) + 1e-8

        return {
            "input_mean": torch.tensor(input_mean, dtype=torch.float32),
            "input_std":  torch.tensor(input_std,  dtype=torch.float32),
            "trans_mean": torch.tensor(trans_mean, dtype=torch.float32),
            "trans_std":  torch.tensor(trans_std,  dtype=torch.float32),
        }

    

    def setup_model(self):
        """Initialize the model based on graph structure."""
        edge_index_sets = []
        
        fc_struc = get_fc_graph_struc(self.feature_map)
        tactile_struc = get_tactile_graph_struc(self.feature_map)

        fc_edge_index = build_loc_net(fc_struc, self.feature_map)
        tactile_edge_index = build_loc_net(tactile_struc, self.feature_map)

        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long, device=self.device)
        tactile_edge_index = torch.tensor(tactile_edge_index, dtype=torch.long, device=self.device)


        if self.config['graph_name'] == 'g':
            edge_index_sets.append(fc_edge_index)
        elif self.config['graph_name'] == 'g+':
            edge_index_sets.append(tactile_edge_index)
        else:
            edge_index_sets.extend([fc_edge_index, tactile_edge_index])
        
        #print(f"feat map: {len(self.feature_map)}")
        

        self.model = TactileGAT(edge_index_sets, len(self.feature_map),
                        graph_name=self.config['graph_name'],
                        dim=self.config['dim'],
                        input_dim=self.config['slide_win'],
                        out_layer_num=self.config['out_layer_num'],
                        out_layer_inter_dim=self.config['out_layer_inter_dim'],
                        topk=self.config['topk']).to(self.device)
        

        
    def myrun(self):
        learned_graphs = []
        #losses = []

        '''
            This for loop works when each item in the dict is one whole dataset. But what happens when the item is actually several subsets?
            The idea is that each item is a list with at least 1 dataset. So its actually a nested for loop where the outer for loop runs
            over items and the inner for loop is the length of the list for that item. Sometimes it can be 1, or other times it could be more, depending on
            how much the trial had to be broken up into subsets.  

        '''

        #train_sampler = RandomSampler(self.train_dataset)
        #val_sampler = RandomSampler(self.val_dataset)

        train_loader = DataLoader(self.train_dataset, batch_size=self.config["batch"])
        val_loader = DataLoader(self.val_dataset, batch_size=self.config["batch"])

        train(self.model, self.config, train_loader, val_loader)

        """Run training and testing."""
        torch.save(self.model.state_dict(), "EGNN6.pth")
        #self.config['model_save_path'], self.config['result_save_path'] = self.get_save_path()
        #train(self.model, self.config, self.train_dataloader, self.val_dataloader)
        #self.model.load_state_dict(torch.load('test2_learngrad.pth'))
        #test_result, total_loss, avg_loss = test(self.model, self.test_dataloader, config=self.config)
        #print(f"average loss: {avg_loss}")
        #learned_graph = self.model.get_learned_graph()

    
        #self.save_results(self.config['result_save_path'], learned_graphs)
    

    def histweight(self, state_dict):
        pass

    def run(self):

        """Run training and testing."""

        # For-loop for train takes in separate train_dataloaders where a subset is either from a different trial, 
        # or its a continuous set from the same trial 

        #self.config['model_save_path'], self.config['result_save_path'] = self.get_save_path()
        train(self.model, self.config, self.train_loader, self.val_loader)
        #self.model.load_state_dict(torch.load(self.config['model_save_path']))
        #test_result, avg_loss, accuracy_rate, precision, recall, f1 = test(self.model, self.test_dataloader)
        #learned_graph = self.model.get_learned_graph()
        #self.save_results(self.config['result_save_path'], accuracy_rate, test_result, learned_graph)
    
    def mytest(self,dataloader):
        filepath = "test3_cleandata_g+.pth"
        model = torch.load(filepath)
        test_data = construct_data(input_output_pairs[4], config=self.config)  # Should return list of dataset(s)
        _, avg_loss = test(model, config, dataloader)


    def get_save_path(self, feature_name=''):
        dir_path = self.config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m-%d_%H-%M-%S')
        datestr = self.datestr
        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pty',
            f'./results/{dir_path}/{datestr}',
        ]
        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)
        return paths[0], paths[1]
    
    def save_results(self, result_save_path, learned_graph):
        """Save test results and learned graph."""
        np_name = f"{result_save_path}_{self.config['topk']}_{self.config['graph_name']}.npy"
        np.save(np_name, {
            'learned_graph': learned_graph
        })

def parse_args():
    parser = argparse.ArgumentParser(description="Main script parameters")
    parser.add_argument('-batch', help='Batch size', type = int, default=256)
    parser.add_argument('-epoch', help='Number of training epochs', type = int, default=50)
    parser.add_argument('-signal_name', help='signal name', type = str, default='')
    parser.add_argument('-graph_name', help='graph name(g or g+ or null for both)', type = str, default='g+')
    parser.add_argument('-slide_win', help='slide_win', type = int, default=200)
    parser.add_argument('--learn_grad', help='learn_grad', action=argparse.BooleanOptionalAction)
    parser.add_argument('-dim', help='dimension', type = int, default=64)#64
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=200)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='tactile')
    parser.add_argument('-model_save_path', help='save model path', type = str, default='')
    parser.add_argument('-result_save_path', help='save result path', type = str, default='')
    parser.add_argument('-dataset', help='Name of Dataset', type = str, default='tactile')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-lr', help='learning rate', type = float, default=0.01)
    parser.add_argument('-max_norm', help='gradient clipping arg', type = float, default=0.01)
    parser.add_argument('-rot_weight', help='rotation weight', type = float, default=1.0)
    parser.add_argument('-trans_weight', help='translation weight', type = float, default=1.0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.3)
    parser.add_argument('-topk', help='topk num', type = int, default=6)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-model_type', help='EGNN or GAT', type = str, default='EGNN')

    #parser.add_argument('rest_state', help='add rest state feature', type=str, default='n')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'signal_name': args.signal_name,
        'graph_name': args.graph_name,
        'slide_win': args.slide_win,
        'learn_grad': args.learn_grad,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'lr': args.lr,
        'max_norm': args.max_norm,
        'rot_weight': args.rot_weight,
        'trans_weight': args.trans_weight,
        'save_path': args.save_path_pattern,
        'model_save_path': args.model_save_path,
        'result_save_path': args.result_save_path,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'model_type': args.model_type
    }
    main = Main(config)
    main.myrun()



'''

Full trial mixing setup_data:

    def setup_data(self):
        """Load and prepare data. Need to figure out how to concatenate total collected data 
        but also make sure the Rest State Vector is a feature that defines each data point.
        All data cannot be treated the same, because data comes from different head shapes/types/weights
        
        Each item should be a list of at least length 1. 1 dataset if it never had to be broken up, or more than length 1 where each element
        is a subset of the trial. If there subsets of a trial, then each subset goes through TimeDataset. 

        The dictionary has to be restructured here a little bit. It should be Trial -> [train], [val]
        
        """
        input_output_pairs = np.array(matching_io(self.path + "/sc",self.path + "/capi"))  # List of matching input/output file paths
        
        feature_map = get_feature_map()
        self.feature_map = feature_map

        self.trials = {}
        self.train_sets = {"train": [],"val": []}

        self.stats_trials = []
    
        for i in range(len(input_output_pairs)):
            train_data = construct_data(input_output_pairs[i])  # Should return list of dataset(s)

            train_dataset = TimeDataset(train_data, mode='train', config=self.config)

            val_size = int(self.config['val_ratio'] * len(train_dataset))
            train_set, val_set = random_split(train_dataset, [len(train_dataset)-val_size, val_size])
            train_dataloader = DataLoader(train_set, batch_size=self.config['batch'], shuffle=True)
            val_dataloader = DataLoader(val_set, batch_size=self.config['batch'], shuffle=False)

            self.train_sets["train"].append(train_dataloader)
            self.train_sets["val"].append(val_dataloader) 

        for i in range(len(input_output_pairs)):
            train_data = construct_data(input_output_pairs[i], config=self.config)  # Should return list of dataset(s)
            trial_key = f"trial_{i}"
            self.trials[trial_key] = {"train": [], "val": []}            
            
            for entry in train_data:
                data = TimeDataset(entry, mode='train', config=self.config) # Window the subset of the trial
                val_size = int(self.config['val_ratio'] * len(data))
                train_set, val_set = random_split(data, [len(data)-val_size, val_size], torch.Generator().manual_seed(42))    # Split into val 

                self.trials[trial_key]["train"].append(train_set)
                self.trials[trial_key]["val"].append(val_set)
        

        This is for random selected windows OVER ALL TRIALS
        s
        all_train_sets = []
        all_val_sets = []

        for trial_key, datasets in self.trials.items():
            all_train_sets.extend(datasets["train"])
            all_val_sets.extend(datasets["val"])

        self.train_dataset = ConcatDataset(all_train_sets)
        self.val_dataset = ConcatDataset(all_val_sets)
                
        for trial_key, datasets in self.trials.items():
            print(f"\n{trial_key}:")
            for split, loaders in datasets.items():
                print(f"  {split}: {len(loaders)} loaders")
                for idx, loader in enumerate(loaders):
                    dataset = loader.dataset
                    print(f"    Loader {idx}: dataset size = {len(dataset)}")
        

'''