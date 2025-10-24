import time
import torch
import torch.nn as nn
import numpy as np
from mytest import test
from utils.device import get_device
from matplotlib import pyplot as plt

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

mle_loss = nn.MSELoss()

def QuaternionLoss(pred, target, rot_weight, trans_weight):

    """
    pred:     [batch, 7]
    target:   [batch, 7] 
    """

    pred_rotation = pred[:, :4]
    pred_translation = pred[:, 4:]

    target_rotation = target[:, :4]
    target_translation = target[:, 4:]

    rotation_loss = quaternion_angular_loss(pred_rotation, target_rotation)
    translation_loss = mle_loss(pred_translation, target_translation)

    #print(f"rotation: {rotation_loss}\ntranslation: {translation_loss}")

    loss = (rot_weight * rotation_loss) + (trans_weight * translation_loss)
    
    return translation_loss, rotation_loss, loss

def quaternion_angular_loss(q1, q2):
    # Mean Absolute Error Implementation
    # Normalize quaternions
    q1 = q1 / q1.norm(p=2, dim=-1, keepdim=True)
    q2 = q2 / q2.norm(p=2, dim=-1, keepdim=True)

    dot_product = (q1 * q2).sum(dim=-1).abs()
    angular_error = 2 * torch.acos(torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7))
    
    return angular_error.mean()

def train_epoch(model, config, dataloader, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    rot_loss_total = 0
    trans_loss_total = 0
    
    for w, x, target in dataloader:
        w, target = [item.float().to(device) for item in [w, target]]
        # w.shape = torch.Size([129, 12, 200])
        optimizer.zero_grad()
        outputs = model(w).float()
        trans_loss, rot_loss, loss = QuaternionLoss(outputs, target, rot_weight=config['rot_weight'], trans_weight=config['trans_weight'])   # Need to make sure this works
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
        optimizer.step()
        total_loss += loss.item()
        rot_loss_total += rot_loss.item()
        trans_loss_total += trans_loss.item()
        #trans_loss_total += trans_loss.item()
        #total_correct += (outputs.argmax(dim=1) == attack_labels).sum().item()
    return trans_loss_total, rot_loss_total, total_loss 

    

def validate(model, config, dataloader, device):
    model.eval()
    with torch.no_grad():
        return test(model, config, dataloader)

def train(model, config, train_dataloader, val_dataloader):
    device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])

    best_accuracy = 0
    patience, trials = 10, 0

    train_losses = []
    val_losses = []
    rot_list = []
    trans_list = []

    for epoch in range(config['epoch']):
        trans_loss, rot_loss, train_loss = train_epoch(model, config, train_dataloader, optimizer, device)
        #total_val_loss, val_loss = validate(model=model, config=config, dataloader=val_dataloader, device=device)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epoch']}, Train Loss: {train_loss:.4f} = Trans_loss {trans_loss} + Rot_loss {rot_loss}")
        
        if not epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{config['epoch']}, Train Loss: {train_loss:.4f}")
        
        else: 
            print(f"Epoch {epoch+1}/{config['epoch']}, Train Loss: {train_loss:.4f}, val loss: {rot_loss:.4f}") #, Trans Loss: {trans_loss:.4f}")

        train_losses.append(train_loss)
        rot_list.append(rot_loss)
        #val_losses.append(val_loss)
    
    # visualization only if GAT has learned attention weights
    if hasattr(model, 'get_learned_graph'):
        graph = model.get_learned_graph()
        if graph is not None:
            edge_index = model.edge_index_sets[0]
            edge_index_np = edge_index.t().cpu().numpy()
            att_weight_np = graph.squeeze().detach().cpu().numpy()
            if att_weight_np.ndim > 1:
                att_weight_np = att_weight_np.mean(axis=-1)
            G = nx.Graph()
            for i, (src, dst) in enumerate(edge_index_np):
                G.add_edge(src, dst, weight=float(att_weight_np[i]))
            pos = nx.spring_layout(G, seed=42)
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            nx.draw(G, pos, with_labels=True, node_color='skyblue',
                    edge_color=weights, width=[w * 5 for w in weights],
                    edge_cmap=plt.cm.Blues)
            plt.show()
    else:
        print("No attention graph to visualize (EGNN uses distance-based messages).")

    plt.figure()
    l = np.array(train_losses)
    r = np.array(rot_list)
    t = np.array(trans_list)

    plt.plot(np.linspace(1, len(l),len(l)),l,'r-')
    plt.title("total loss")
    
    plt.plot(np.linspace(1, len(r),len(r)),r,'g-')
    plt.title("rotation loss")


    plt.figure()
    plt.plot(np.linspace(1, len(t),len(t)),t,'b-')
    plt.title("trans loss")

    plt.show()

