import torch
import numpy as np
import sys
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(targets, predictions, num_classes):
    confusion_mat = confusion_matrix(y_true=targets.flatten(), y_pred=predictions.flatten(), labels=range(num_classes))
    return confusion_mat

def measure_performance(confusion_mat, num_classes, ignore_label=None, smooth=sys.float_info.epsilon):
    ious = []
    classes_with_missing_representatives = []

    for c in range(num_classes):
        if c != ignore_label:
            intersection = confusion_mat[c, c]
            union = confusion_mat[c, :].sum() + confusion_mat[:, c].sum() - intersection

            if confusion_mat[c, :].sum() == 0:
                classes_with_missing_representatives += [c]
                ious.append(0)
            else:
                ious.append((intersection + smooth) / (union + smooth) if union > 0 else 0)
        else:
            ious.append(0)

    miou = np.array(ious).sum()
    ious = np.array(ious)

    if ignore_label != None:
        miou -= ious[ignore_label]
        miou /= (num_classes - 1 - len(classes_with_missing_representatives))
    else:
        miou /= (num_classes - len(classes_with_missing_representatives))

    for c in classes_with_missing_representatives:
        ious[c] = -1

    return miou, ious

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
