import torch
import time
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.device import get_device

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
    
    return rotation_loss, loss

def quaternion_angular_loss(q1, q2):
    # Mean Absolute Error Implementation
    # Normalize quaternions
    q1 = q1 / q1.norm(p=2, dim=-1, keepdim=True)
    q2 = q2 / q2.norm(p=2, dim=-1, keepdim=True)

    dot_product = (q1 * q2).sum(dim=-1).abs()
    angular_error = 2 * torch.acos(torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7))
    
    return angular_error.mean()

def test(model, config, dataloader):
    device = get_device()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_rot_loss = 0.0
    
    all_predicted, all_labels = [], []
    
    for w, x, target in dataloader:
        w, target = [item.float().to(device) for item in [w, target]]
        
        with torch.no_grad():
            outputs = model(w)
            rot_loss, loss = QuaternionLoss(outputs, target, rot_weight=config['rot_weight'], trans_weight=config['trans_weight'])

            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            
        
        #all_predicted.extend(outputs.cpu().numpy())

        avg_loss = total_rot_loss / w.size(1)


    return total_loss, avg_loss
