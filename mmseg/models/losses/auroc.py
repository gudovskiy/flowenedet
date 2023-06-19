import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

softmax = nn.Softmax(dim=1)

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def auroc(linear_logits, flow_margits, target, ood_class_index):
    """Calculate AuROC according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)

    Returns:
        float | tuple[float]: 
    """
    if 1:
        assert linear_logits.ndim == target.ndim + 1
        assert flow_margits.ndim == target.ndim + 1
        assert linear_logits.size(0) == target.size(0)
        assert flow_margits.size(0) == target.size(0)
        device = linear_logits.device
        B, CL, H, W = linear_logits.size()  # batch size
        S = H*W
        E = B*S
        #print(linear_logits.shape, flow_pred.shape, target.shape)
        linear_preds = torch.argmax(linear_logits, dim=1)  # BxHxW softmax prediction
        m = linear_preds.eq(target.view_as(linear_preds)) # BxHxW=E
        idd_label = t2np(~m).reshape(E)  # E .astype(np.uint8)
        y_label = t2np(target).reshape(E)  # classifier groundtruth
        ood_mask = np.isin(y_label, ood_class_index)
        ood_label = np.zeros_like(y_label)  # ood groundtruth
        ood_label[ood_mask] = 1
        cdd_label = ood_label.copy()
        #print(idd_label.shape, ood_mask.shape, ood_label.shape)
        cdd_label[~ood_mask] = idd_label[~ood_mask]
        ours_dist = 1.0 - t2np(flow_margits.reshape(E))
        checksum = np.sum(cdd_label)
        #if checksum != 0 and checksum != cdd_label.size:
        cdd_auroc = torch.tensor(100.0*roc_auc_score(cdd_label, ours_dist), device=device)
    else:
        cdd_auroc = torch.zeros(1, device=device)
    #
    return cdd_auroc


class AuROC(nn.Module):
    """AuROC calculation module."""

    def __init__(self):
        """Module to calculate the accuracy.

        Args:
        """
        super().__init__()

    def forward(self, linear_logits, flow_margits, target, ood_class_index):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return auroc(linear_logits, flow_margits, target, ood_class_index)
