import torch
import sys
from sklearn.metrics import confusion_matrix

def iou(predictions, targets, num_classes, smooth=sys.float_info.epsilon):
    confusion_mat = confusion_matrix(targets.flatten(), predictions.flatten(), labels=range(num_classes))
    ious = []
    for c in range(num_classes):
        intersection = confusion_mat[c, c]
        union = confusion_mat[c, :].sum() + confusion_mat[:, c].sum() - intersection
        ious.append((intersection + smooth) / (union + smooth))
    return ious

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
