# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import random
from sklearn.metrics import precision_recall_curve  # roc_curve, auc
from ood_metrics import auroc, aupr, fpr_at_95_tpr

import os
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300
#figsize = (30*cm, 60*cm)


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        pred_score,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False,
                        f1_thresh=None):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """
    
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(pred_score, str):
        pred_score = torch.from_numpy(np.load(pred_score))
    else:
        pred_score = torch.from_numpy((pred_score))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    
    #intersect_id = pred_label_id[pred_label_id == label_id]
    #area_intersect_id = torch.histc(
    #    intersect_id.float(), bins=(num_classes), min=0, max=num_classes - 1)
    
    if f1_thresh is not None:
        # we reject all predictions from IDM&OOD
        label_mask = label != pred_label  # label IDM/OOD
        label[label_mask] = num_classes-1
        det_mask = (pred_score > f1_thresh)  # predict IDM/OOD
        pred_label[det_mask] = num_classes-1
        # None:
        #det_mask = (pred_score > 1.0)  # predict IDM/OOD
        #pred_label[det_mask] = num_classes-1
        #label_mask = label == ignore_index  # label OOD
        #label[label_mask] = num_classes-1
    else:
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
    
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    
    area_union = area_pred_label + area_label - area_intersect  # - 2*area_intersect + area_intersect_id
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(all_results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False,
                              openIoU=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    
    # openIoU threshold
    if openIoU:
        f1_thresh = f1_threshold(all_results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label)
    else:
        f1_thresh = None
    
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for pred, dist, gt_seg_map in zip(all_results[0], all_results[1], gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                pred, dist, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label, f1_thresh=f1_thresh)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def viz_ood(img, pred_label, pred_dist, label, threshold, stat_dist, ood_class_index, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Load info for OOD metrics.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(pred_dist, str):
        pred_dist = np.load(pred_dist)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    OUT_DIR = './viz'
    MODEL = 'deeplabv3plus.r101.SML'
    CORRUPTION = 'city'
    #CORRUPTION = 'city-b3'
    #CORRUPTION = 'city-b5'
    #CORRUPTION = 'city-s1'
    #CORRUPTION = 'city-s3'
    #CORRUPTION = 'fishy-st'
    #CORRUPTION = 'fishy-lf'
    image_dirs = os.path.join(OUT_DIR, MODEL, CORRUPTION)
    os.makedirs(image_dirs, exist_ok=True)
    # new palette with OOD classes
    CITY_PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
                    [120, 220, 60], [0, 255, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],
                    [119, 11, 32]]
    #
    FISHY_PALETTE = [[0, 0, 153], [204, 0, 0], [0, 0, 0]]
    #
    y_label = label  # classifier groundtruth
    p_label = pred_label # classifier prediction
    d_score = pred_dist  # distance prediction
    # OOD
    ood_mask = np.isin(label, ood_class_index)
    ood_label = np.zeros_like(y_label)  # ood groundtruth
    ood_label[ood_mask] = 1
    if 'fishy' in CORRUPTION:
        # CDD
        cdd_label = ood_label.copy()
        cdd_label[label == ignore_index] = ignore_index
    else:
        # IDM
        idm_mask = (p_label != y_label)  # classifier hits
        idm_label = idm_mask.astype(y_label.dtype)
        # CDD
        cdd_label = ood_label.copy()
        cdd_label[~ood_mask] = idm_label[~ood_mask]
    #
    seg_label = np.uint8(p_label)
    ann_label = np.uint8(y_label)
    cdd_label = np.uint8(cdd_label)
    #
    dist_min, dist_max = stat_dist
    det = d_score.copy() - dist_min
    det/= (dist_max-dist_min)
    det = (255.0*np.clip(det, 0.0, 1.0)).astype(np.uint8)
    #
    seg = np.zeros((seg_label.shape[0], seg_label.shape[1], 3), dtype=np.uint8)
    ann = np.zeros_like(seg)
    cdd = np.zeros_like(seg)
    #
    maskf1 = np.ones_like(cdd) * FISHY_PALETTE[0]
    maskf1[d_score > threshold, :] = FISHY_PALETTE[1]
    cdd[cdd_label == 0, :] = FISHY_PALETTE[0]
    cdd[cdd_label == 1, :] = FISHY_PALETTE[1]
    #
    print(img, threshold, pred_dist.max(), pred_dist.min(), ignore_index)
    maskf1[cdd_label == ignore_index, :]  = FISHY_PALETTE[2]
    cdd[cdd_label == ignore_index, :] = FISHY_PALETTE[2]
    det[cdd_label == ignore_index] = 0
    for cl, color in enumerate(CITY_PALETTE):
        seg[seg_label == cl, :] = color
        ann[ann_label == cl, :] = color
    #
    print(img, image_dirs, seg.shape, ann.shape, det.shape, det.min(), det.max())
    # OOD F1
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(maskf1)
    image_file = os.path.join(image_dirs, '{:08d}_f1m'.format(img))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    # IDM&OOD PREDICT
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(det, cmap='jet', norm=norm, alpha=1.0, interpolation='none')
    image_file = os.path.join(image_dirs, '{:08d}_det'.format(img))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    # IDM&OOD LABEL
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cdd)
    image_file = os.path.join(image_dirs, '{:08d}_cdd'.format(img))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    # SEG PREDICT
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(seg)
    image_file = os.path.join(image_dirs, '{:08d}_seg'.format(img))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    # SEG LABEL
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(ann)
    image_file = os.path.join(image_dirs, '{:08d}_ann'.format(img))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


def load_ood(pred_label, pred_dist, label, num_classes, ood_class_index, ignore_index, label_map=dict(), reduce_zero_label=False, subsample=True):
    """Load info for OOD metrics.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(pred_dist, str):
        pred_dist = np.load(pred_dist)

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    
    mask = (label != ignore_index)
    pred_label = np.uint8(pred_label[mask])
    pred_dist = np.float32(pred_dist[mask])
    label = np.uint8(label[mask])

    '''idm_mask = label != 255
    idm_label = label[idm_mask]
    idm_pred  = pred_label[idm_mask]
    check = random.sample(range(len(idm_pred)), 32)
    print('y:', idm_label[check])
    print('p:', idm_pred[check])'''

    hit = pred_label == label
    mis = pred_label != label
    idx = np.array([], dtype=int)
    idx_all = np.arange(label.size)
    sample_size = 2**16
    if subsample:
        # add OOD indices
        mask_ood = np.isin(label, ood_class_index)
        mask_idm = ~mask_ood
        if np.sum(mask_ood) > 0:
            idx_ood = idx_all[mask_ood]
            if len(idx_ood) > sample_size:
                idx_ood = idx_ood[random.sample(range(len(idx_ood)), sample_size)]
            idx = np.concatenate((idx, idx_ood))
        # add IDM indices
        mask_hit = mask_idm*hit
        if np.sum(mask_hit) > 0:
            idx_hit = idx_all[mask_hit]
            if len(idx_hit) > sample_size//2:
                idx_hit = idx_hit[random.sample(range(len(idx_hit)), sample_size//2)]
            idx = np.concatenate((idx, idx_hit))
        # add IDM indices
        mask_mis = mask_idm*mis
        if np.sum(mask_mis) > 0:
            idx_mis = idx_all[mask_mis]
            if len(idx_mis) > sample_size//2:
                idx_mis = idx_mis[random.sample(range(len(idx_mis)), sample_size//2)]
            idx = np.concatenate((idx, idx_mis))
        
        '''for cl in range(num_classes):
            mask_sub = np.isin(label, cl)
            if np.sum(mask_sub) > 0:
                idx_sub = idx_all[mask_sub]
                if len(idx_sub) > sample_size:
                    idx_sub = idx_sub[random.sample(range(len(idx_sub)), sample_size)]
                idx = np.concatenate((idx, idx_sub))'''
        
        sub_pred_label = pred_label[idx]
        sub_pred_dist = pred_dist[idx]
        sub_label = label[idx]
        '''check = random.sample(range(len(sub_label)), 32)
        print('y:', sub_label[check])
        print('p:', sub_pred_label[check])
        print('dst:', sub_pred_dist[check])'''
        
        return sub_pred_label, sub_pred_dist, sub_label
        
    else:
        return pred_label, pred_dist, label


def fishy_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Total Area under ROC.
    """
    d_list = list()
    y_list = list()
    num_imgs = len(gt_seg_maps)
    for pred_label, pred_dist, label in zip(results[0], results[1], gt_seg_maps):
        _, dist_i, label_i = load_ood(pred_label, pred_dist, label, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label, subsample=False)
        d_list.extend(dist_i)
        y_list.extend(label_i)
    #
    y_label = np.asarray(y_list)  # classifier groundtruth
    d_score = np.asarray(d_list)  # distance prediction
    # OOD
    ood_mask = np.isin(y_label, ood_class_index)
    ood_label = np.zeros_like(y_label)  # ood groundtruth
    ood_label[ood_mask] = 1
    # sklearn
    precision, recall, thresholds = precision_recall_curve(ood_label, d_score)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    ood_threshold = thresholds[np.argmax(f1)]
    print('\nOptimal OOD Threshold: {:.2f}'.format(ood_threshold))
    # ood_metrics
    ood_det_roc_auc = 100.0*auroc(d_score, ood_label)
    ood_det_pr_auc  = 100.0*aupr(d_score, ood_label)
    ood_det_fpr_95  = 100.0*fpr_at_95_tpr(d_score, ood_label)
    print('\nAUROC/AUPR/FPR95 OOD: {:.2f}, {:.2f}, {:.2f}'.format(ood_det_roc_auc, ood_det_pr_auc, ood_det_fpr_95))
    # per-class is not needed for Fishyscapes
    total_auroc = torch.zeros(num_classes, dtype=torch.float64)
    total_auroc[0] = ood_det_roc_auc/100.0
    total_auroc[1] = ood_det_pr_auc/100.0
    total_auroc[2] = ood_det_fpr_95/100.0
    #
    viz = False  # True False  # to generate qualitative Figures in the paper
    if viz:
        stat_dist = d_score.min(), d_score.max()
        for i in range(num_imgs):  # need for efficient implementation due to memory size
            viz_ood(i, results[0][i], results[1][i], gt_seg_maps[i], ood_threshold, stat_dist, ood_class_index, ignore_index, label_map, reduce_zero_label)
    #
    return total_auroc


def idm_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Total Area under ROC.
    """
    idm_det_roc_auc, idm_det_pr_auc, idm_det_fpr_95, idm_threshold = 0.0, 0.0, 0.0, 0.0
    num_imgs = len(gt_seg_maps)
    for pred_label, pred_dist, label in zip(results[0], results[1], gt_seg_maps):
        pred_i, dist_i, label_i = load_ood(pred_label, pred_dist, label, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label, subsample=False)
        #
        y_label = np.asarray(label_i)  # classifier groundtruth
        p_label = np.asarray(pred_i)  # classifier prediction
        d_score = np.asarray(dist_i)  # distance prediction
        # IDM
        idm_mask = (p_label != y_label)  # classifier hits
        idm_label = idm_mask.astype(y_label.dtype)
        # sklearn
        precision, recall, thresholds = precision_recall_curve(idm_label, d_score)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        # idm_metrics
        idm_threshold   += thresholds[np.argmax(f1)]
        idm_det_roc_auc += 100.0*auroc(d_score, idm_label)
        idm_det_pr_auc  += 100.0*aupr(d_score, idm_label)
        idm_det_fpr_95  += 100.0*fpr_at_95_tpr(d_score, idm_label)
        #print('\nAUROC/AUPR/FPR95 IDM: {:.2f}, {:.2f}, {:.2f}'.format(100.0*auroc(d_score, idm_label), 100.0*aupr(d_score, idm_label), 100.0*fpr_at_95_tpr(d_score, idm_label)))
    #
    idm_det_roc_auc /= num_imgs
    idm_det_pr_auc  /= num_imgs
    idm_det_fpr_95  /= num_imgs
    idm_threshold   /= num_imgs
    print('\nOptimal IDM Threshold: {:.2f}'.format(idm_threshold))
    print('\nAUROC/AUPR/FPR95 IDM: {:.2f}, {:.2f}, {:.2f}'.format(idm_det_roc_auc, idm_det_pr_auc, idm_det_fpr_95))
    # per-class is not needed for Fishyscapes
    total_auroc = torch.zeros(num_classes, dtype=torch.float64)
    total_auroc[0] = idm_det_roc_auc/100.0
    total_auroc[1] = idm_det_pr_auc/100.0
    total_auroc[2] = idm_det_fpr_95/100.0
    #
    viz = False  # True False  # to generate qualitative Figures in the paper
    if viz:
        stat_dist = d_score.min(), d_score.max()
        for i in range(num_imgs):  # need for efficient implementation due to memory size
            viz_ood(i, results[0][i], results[1][i], gt_seg_maps[i], idm_threshold, stat_dist, ood_class_index, ignore_index, label_map, reduce_zero_label)
    #
    return total_auroc

def f1_threshold(results, gt_seg_maps, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate F1 Threshold
    """
    f1_thresholds = 0.0
    num_imgs = len(gt_seg_maps)
    modified_ignore_index = -100  # such that we load OOD areas as well
    for pred_label, pred_dist, label in zip(results[0], results[1], gt_seg_maps):
        pred_i, dist_i, label_i = load_ood(pred_label, pred_dist, label, num_classes, modified_ignore_index, modified_ignore_index, label_map, reduce_zero_label, subsample=False)
        #
        y_label = np.asarray(label_i)  # classifier groundtruth
        p_label = np.asarray(pred_i)  # classifier prediction
        d_score = np.asarray(dist_i)  # distance prediction
        # IDM&OOD
        f1_label = (p_label != y_label).astype(y_label.dtype)
        # sklearn
        precision, recall, thresholds = precision_recall_curve(f1_label, d_score)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        f1_threshold   = thresholds[np.argmax(f1)]
        f1_thresholds += f1_threshold
        #print('\n F1 threshold: {:.2f}'.format(f1_threshold))
    #
    f1_thresholds /= num_imgs
    print('\nOptimal F1 threshold: {:.6f}'.format(f1_thresholds))
    return f1_thresholds


def our_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Total Area under ROC.
    """
    p_list = list()
    d_list = list()
    y_list = list()
    num_imgs = len(gt_seg_maps)
    if ignore_index in ood_class_index:
        ignore_index = 250  # background is OOD class in our metrics!
    
    for pred_label, pred_dist, label in zip(results[0], results[1], gt_seg_maps):
        pred_i, dist_i, label_i = load_ood(pred_label, pred_dist, label, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label, subsample=True)
        p_list.extend(pred_i)
        d_list.extend(dist_i)
        y_list.extend(label_i)
    #
    y_label = np.asarray(y_list)  # classifier groundtruth
    p_label = np.asarray(p_list)  # classifier prediction
    d_score = np.asarray(d_list)  # distance prediction
    # OOD
    ood_mask = np.isin(y_label, ood_class_index)
    ood_label = np.zeros_like(y_label)  # ood groundtruth
    ood_label[ood_mask] = 1
    # IDM
    idm_mask = (p_label != y_label)  # classifier hits
    idm_label = idm_mask.astype(y_label.dtype)
    idm_label[ood_mask] = 0
    # CDD
    cdd_label = ood_label.copy()
    cdd_label[~ood_mask] = idm_label[~ood_mask]
    #
    viz = False  # True  # to generate qualitative Figures in the paper
    if viz:
        # sklearn
        precision, recall, thresholds = precision_recall_curve(cdd_label, d_score)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cdd_threshold = thresholds[np.argmax(f1)]
        print('Optimal CDD Threshold: {:.2f}'.format(cdd_threshold))
        stat_dist = d_score.min(), d_score.max()
        for i in range(num_imgs):  # need for efficient implementation due to memory size
            if i in [1,2,3,13,17,39,41,43,89,97]:  # [13,17]: #< 100:
                viz_ood(i, results[0][i], results[1][i], gt_seg_maps[i], cdd_threshold, stat_dist, ood_class_index, ignore_index, label_map, reduce_zero_label)
    # metrics
    idm_det_roc_auc = 100.0*auroc(d_score[~ood_mask], idm_label[~ood_mask])
    ood_det_roc_auc = 100.0*auroc(d_score, ood_label)
    cdd_det_roc_auc = 100.0*auroc(d_score, cdd_label)
    idm_det_pr_auc  = 100.0*aupr(d_score[~ood_mask], idm_label[~ood_mask])
    ood_det_pr_auc  = 100.0*aupr(d_score, ood_label)
    cdd_det_pr_auc  = 100.0*aupr(d_score, cdd_label)
    idm_det_fpr_95  = 100.0*fpr_at_95_tpr(d_score[~ood_mask], idm_label[~ood_mask])
    ood_det_fpr_95  = 100.0*fpr_at_95_tpr(d_score, ood_label)
    cdd_det_fpr_95  = 100.0*fpr_at_95_tpr(d_score, cdd_label)
    print('\nAUROC/AUPR/FPR95 IDM/OOD/CDD: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
        idm_det_roc_auc, ood_det_roc_auc, cdd_det_roc_auc,\
        idm_det_pr_auc, ood_det_pr_auc, cdd_det_pr_auc,\
        idm_det_fpr_95, ood_det_fpr_95, cdd_det_fpr_95))
    
    # per-class
    #for cl in range(num_classes):
    #    if cl in ood_class_index:
    #        checksum = np.sum(ood_label)
    #        if checksum != 0 and checksum != ood_label.size:
    #            total_auroc[cl] = auroc(ood_score, ood_label)
    #    else:
    #        idm_mask = np.isin(idm_label, cl)
    #        checksum = np.sum(idm_label[idm_mask])
    #        if checksum != 0 and checksum != idm_label[idm_mask].size:
    #            total_auroc[cl] = auroc(idm_score[idm_mask], idm_label[idm_mask])
    
    total_auroc = torch.zeros(num_classes, dtype=torch.float64)
    total_auroc[0] = idm_det_roc_auc/100.0
    total_auroc[1] = ood_det_roc_auc/100.0
    total_auroc[2] = cdd_det_roc_auc/100.0
    total_auroc[3] = idm_det_pr_auc/100.0
    total_auroc[4] = ood_det_pr_auc/100.0
    total_auroc[5] = cdd_det_pr_auc/100.0
    total_auroc[6] = idm_det_fpr_95/100.0
    total_auroc[7] = ood_det_fpr_95/100.0
    total_auroc[8] = cdd_det_fpr_95/100.0
    #
    return total_auroc


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ood_class_index,
                 ignore_index,
                 metrics=['mIoU', 'mOurs'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ood_class_index (list[int]): The labels indices to be used for OOD metrics. Default: None
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU', 'mDice', 'mOurs', 'mFishy', 'oIoU'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    openIoU = True if 'oIoU' in metrics else False
    list_gt_seg_maps = list(gt_seg_maps)  # need to convert a generator to list because of multiple uses
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, list_gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label, openIoU=openIoU)
    ret_metrics = total_area_to_metrics(results, list_gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label,
                                        total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(all_pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(all_pre_eval_results, tuple):
        pre_eval_results = all_pre_eval_results[0]
    else:
        pre_eval_results = all_pre_eval_results
    
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(all_pre_eval_results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label,
                                        total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label,
                          total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU', 'mOurs'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mOurs', 'mFishy', 'oIoU']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        elif metric == 'mOurs':
            #ood_metrics = our_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label)
            ood_metrics = idm_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label)
            ret_metrics['Ours'] = ood_metrics
        elif metric == 'mFishy':
            ood_metrics = fishy_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label)
            ret_metrics['Fishy'] = ood_metrics
        elif metric == 'oIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['oIoU'] = iou
            ret_metrics['oAcc'] = acc

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


'''
def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ood_class_index,
                 ignore_index,
                 metrics=['mIoU', 'mOurs'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ood_class_index (list[int]): The labels indices to be used for OOD metrics. Default: None
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU', 'mDice', 'mOurs', 'mFishy'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    allowed_metrics = ['mIoU', 'mDice', 'mOurs', 'mFishy']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    if 'mIoU' in metrics or 'mDice' in metrics:
        total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results[0], gt_seg_maps, num_classes, ignore_index, label_map,
                reduce_zero_label)
        ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                            total_area_pred_label,
                                            total_area_label, metrics, nan_to_num,
                                        beta)
    else:
        ret_metrics = [0.0, np.zeros((num_classes,), dtype=np.float)]
    
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
        elif metric == 'mOurs':
            ood_metrics = our_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label)
            ret_metrics.append(ood_metrics)
        elif metric == 'mFishy':
            ood_metrics = fishy_metrics(results, gt_seg_maps, num_classes, ood_class_index, ignore_index, label_map, reduce_zero_label)
            ret_metrics.append(ood_metrics)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    
    return ret_metrics
'''
