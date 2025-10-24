import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU, Dropout, Embedding
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F

from myutils import get_batch_edge_index

'''
nn.Linear(256, 256),
nn.LayerNorm(256),
nn.ReLU(),
nn.Dropout(0.2),
'''

class FNN(nn.Module):
    def __init__(self, Nd_input, rotation_weight=1.0, translation_weight=1.0):
        super(FNN, self).__init__()

        self.Nd_input = Nd_input    # Number of input dimensions + rest_state
        #self.Nd_target = Nd_target    # Number of target dimensions
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

        self.fc1 = nn.Sequential(
            nn.Linear(self.Nd_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 7)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        out = self.fc1(x)
        return out

def loop_closure_loss(p, q, windows):
    # p: [B, T, 3]  positions; q: [B, T, 4] quats; windows: list of (t1, t2)
    loss = p.new_zeros(())
    for (t1, t2) in windows:
        loss = loss + (p[:, t2] - p[:, t1]).norm(dim=-1).mean()
        # orientation geodesic (optional)
        inner = (q[:, t2] * q[:, t1]).abs().sum(dim=-1).clamp(0, 1)
        loss = loss + (2.0 * torch.arccos(inner)).mean()
    return loss

# This creates blocks of {Linear, Batch, Relu} 
class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):

            if i == layer_num - 1:  # Last layer that is a Linear of shape: [in_num,1] or [inter_num,1] if layer_num is greater than 1
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
    
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        
        self.mlp = nn.ModuleList(modules)
        print("Out-Layer Architecture")
        for i, layer in enumerate(self.mlp):
            print(f"Layer {i}: {layer}")

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d): # if mod is an instance of nn.BatchNorm1D which is the last layer of the module generated above
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        
        return out


# Replace the OUT layer with a more pose friendly MLP

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, node_num=11, bias=True, inter_dim=1, **kwargs):
        super(GraphLayer, self).__init__(aggr='sum', node_dim=0, **kwargs)

        '''
        in_channels: node_num
        out_channels: dim of feature space
        '''

        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=True)    # First projects 

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):

        
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x) 
        else:
            x = (self.lin(x[0]), self.lin(x[1]))
    
        
        edge_index, _ = remove_self_loops(edge_index=edge_index)
        edge_index, _ = add_self_loops(edge_index=edge_index,
                                       num_nodes=x[1].size(self.node_dim))
        
        
        out = self.propagate(edge_index=edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)
        
        if self.concat:

            out = out.view(-1, self.heads * self.out_channels)
        
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out
    
    def message(self, x_i, x_j, edge_index_i, size_i, 
                embedding,
                edges,
                return_attention_weights):
        
        
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)


        if embedding is not None:            
            # Chat's "debug"
            embedding_i = embedding[edges[1] % self.node_num]
            embedding_j = embedding[edges[0] % self.node_num]
            #embedding_i, embedding_j = embedding[edges[1]], embedding[edges[0]]

            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat([x_i, embedding_i], dim=-1)
            key_j = torch.cat([x_j, embedding_j], dim=-1)

            cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
            cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        
        else:
            key_i = x_i
            key_j = x_j

            cat_att_i = self.att_i
            cat_att_j = self.att_j

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)           
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)    # Pass alpha through relu with negative slope
        
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)    # softmax original line

        if return_attention_weights:
            self.__alpha__ = alpha
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        w = x_j * alpha.view(-1, self.heads, 1)
        
        return w  # Aggregation of weighted messages
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GATLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GATLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, node_num=node_num,inter_dim=inter_dim, 
                              heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)   

        out = self.bn(out)
        out = self.leaky_relu(out)
        return out, att_weight

    
class TactileGAT(nn.Module):
    def __init__(self, edge_index_sets, node_num, graph_name=False, dim=64, out_layer_inter_dim=256, input_dim=11, out_layer_num=1, topk=20):
        super(TactileGAT, self).__init__()
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.graph_name = graph_name
        self.dim = dim
        self.device = None
        self.topk = topk
        self.embedding = Embedding(node_num, dim)   # [11, 64] Linear combination layer
        self.bn_outlayer_in = BatchNorm1d(dim * 2 if graph_name == 'gg' else dim)
        
        self.gnn_layers = nn.ModuleList([
            GATLayer(input_dim, dim, node_num= self.node_num, inter_dim=dim+dim, heads=1) for _ in range(1) # Input 
        ])

        self.out_layer = OutLayer(dim * len(edge_index_sets), node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.FNN_layer = FNN(Nd_input=dim*self.node_num)
        
        #self.FNN_layer2 = nn.ModuleList(FNN(Nd_input=dim * len(edge_index_sets)) for _ in )

        self.out_lin = Linear(node_num, 4)
        self.dropout = Dropout(0.2)
        self.cache_edge_index_sets = [None] * len(edge_index_sets)
        self.init_params()
        self.learned_graph = None

        #print(f"initial node_num: {self.node_num}") = 11
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
    
    def forward(self, data):    # CAREFUL HOW X IS FORMATTED BEFOREHAND !!!
        x, self.device = data.clone().detach(), data.device

        batch_num = x.size(0)   # [batch, Nd_input, time]
        x = x.view(-1, x.size(-1))  # [batch*Nd_input, time] | basically collapses the batch into 2D tensor but still keeps track

        gat_outputs = []
        attention_weights_list = []
        #print(len(self.edge_index_sets))
        #for i, edge_index in enumerate(self.edge_index_sets):
        edge_index = self.edge_index_sets[0]
        for i, edge_index in enumerate(self.edge_index_sets):
            
            batch_edge_index = self.prepare_batch_edge_index(i, edge_index, batch_num)
            gat_out, attn_weights = self.gnn_layers[i](x, batch_edge_index, node_num=self.node_num * batch_num, embedding=self.embedding.weight) # Pass through GAT
            gat_outputs.append(gat_out)
            attention_weights_list.append(attn_weights)
        
        self.learned_graph = torch.cat(attention_weights_list, dim=0)   # Attention weight concatenated should be the same dimension as the graph
        
        '''
        gat_ouput[0] = torch.Size([2816, 64]) 2816 = node_num * batch_num
        batch_num: 256
        node_num: 11
        dim: 64
        '''

        #x = torch.mean(gat_outputs,dim=1)
        x = torch.cat(gat_outputs, dim=1)   
        x = x.view(batch_num, self.node_num, -1)    # [batch_num, node_num, dim]
        #x = x.view(x.size(0), -1)  # flatten to [batch_num, 704]
        ''' There is sum, max, and mean pooling. Try different ones'''
        #x = x.mean(dim=1)         # [B, 64], average node features
        #x = x.max(dim=1).values
        #x = x.sum(dim=1)

        #x = self.FNN_layer(x)           # [B, 4]
        #x = x / x.norm(dim=-1, keepdim=True)  # normalize quaternion

        return x
    
    

     
    def prepare_batch_edge_index(self, i, edge_index, batch_num):
        # Debug statement if the cached graph is None or not equal to a batch of the graph
        if self.cache_edge_index_sets[i] is None or self.cache_edge_index_sets[i].size(1) != edge_index.size(1)*batch_num:
            self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, self.node_num).to(self.device)
        return self.cache_edge_index_sets[i]
    
    def process_output(self, x):
        x = F.relu(self.bn_outlayer_in(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.dropout(self.out_layer(x)) # Process learned graph attention 
        
        return x
    
    def my_process(self, x):
        x = F.relu(self.bn_outlayer_in(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.dropout(self.FNN_layer(x)) # Process learned graph attention 
        return x
    
    def get_learned_graph(self):
        return self.learned_graph





   