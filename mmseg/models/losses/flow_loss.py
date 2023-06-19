import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

actSoft     = nn.Softmax(dim=1)
actLogSoft  = nn.LogSoftmax(dim=1)


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def flow_loss(logit, label, num_classes, alpha, beta,
                weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=None):
    
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss_con = F.cross_entropy(logit, label, reduction='none', ignore_index=ignore_index)
    loss_con = weight_reduce_loss(loss_con, weight=weight, reduction=reduction, avg_factor=avg_factor)
    if ignore_index == -101:
        logit_bin = torch.cat((
            torch.logsumexp(logit[:,:-1,...], dim=1, keepdim=True), 
            logit[:,-1,...].unsqueeze(1)), dim=1)
        label_bin = 1*(label >= num_classes)
        loss_bin = F.cross_entropy(logit_bin, label_bin, reduction='none')
        loss_bin = weight_reduce_loss(loss_bin, reduction=reduction, avg_factor=avg_factor)
    else:
        loss_bin = torch.zeros_like(loss_con)
    return loss_con, loss_bin


def flow_binary_loss(linear_logit, flow_logit, label, ood_model, num_classes, alpha, beta,
                weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=None):

    CL = num_classes
    if   'FMD' in ood_model:
        y_mix = torch.zeros_like(label)
    elif 'GMM' in ood_model:
        y_mix = label
    elif 'EXP' in ood_model:
        argSoft = torch.argmax(linear_logit, dim=1)
        y_mix = argSoft.clone()
        mSoft = (label != argSoft)  # miss/hit
        if torch.sum(mSoft) != 0:  # make sure we have negatives
            y_mix[mSoft] = y_mix[mSoft] + CL
    
    label, weight = _expand_onehot_labels(y_mix, weight, flow_logit.shape, ignore_index)
    #label_shadow_one_hot = F.one_hot(label_shadow, num_classes=2*num_classes).float()
    
    loss = F.binary_cross_entropy_with_logits(flow_logit, label.float(), reduction='none')
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    # do the reduction for the weighted loss
    loss = alpha * weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    #loss_cond[loss_cond != loss_cond] = 0.0  # Replace NaN's with 0
    return loss


@LOSSES.register_module()
class FlowLoss(nn.Module):
    """FlowLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, num_classes=-1, alpha=1.0, beta=1.0, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0):
        super(FlowLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        #elif self.use_mask:
        #    self.cls_criterion = mask_flow_loss
        if self.use_sigmoid:
            self.cls_criterion = flow_binary_loss
        else:
            self.cls_criterion = flow_loss

    def forward(self, logit, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if 0:  #self.class_weight is not None:
            print(self.class_weight, logit.shape)
            class_weight = torch.tensor(self.class_weight).to(logit.device)
        else:
            class_weight = None
        losses = self.cls_criterion(logit, label, self.num_classes, self.alpha, self.beta,
            weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        
        #losses = [self.loss_weight * loss for loss in losses]
        return losses
