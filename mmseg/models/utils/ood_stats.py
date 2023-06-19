import os, tempfile
import numpy as np
import random
import torch
import torch.nn.functional as F

REG = 1e-3

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def save_stats(class_means, class_vars, label, logit, dir, num_classes, ignore_index, momentum=1e-2):
    device = class_means.device
    CL = num_classes
    B, C, H, W = logit.shape
    m = momentum
    assert CL == C
    label = t2np(label.squeeze(1))
    logit = t2np(logit)
    pVal, pIdx = np.amax(logit, axis=1), np.argmax(logit, axis=1)
    for c in range(C):
        #mask = np.isin(pIdx, c)
        mask = np.isin(pIdx, c) * np.isin(label, c)
        if np.sum(mask) > 1:
            class_mean = torch.from_numpy(np.asarray(np.mean(pVal[mask]))).float().to(device)
            class_var  = torch.from_numpy(np.asarray(np.var( pVal[mask]))).float().to(device)
            if class_means[c] == 0.0 and class_vars[c] == 1.0:
                class_means[c] = class_mean
                class_vars[c]  = class_var
            else:
                class_means[c] = (1-m)*class_means[c] + m*class_mean
                class_vars[c]  = (1-m)*class_vars[ c] + m*class_var
    #
    train_stats_file = dir + '_stats.npz'
    print('Saving to {}'.format(train_stats_file))
    np.savez(train_stats_file, class_means=t2np(class_means), class_vars=t2np(class_vars))


def load_stats(dir):
    train_stats_file = dir + '_stats.npz'
    if os.path.isfile(train_stats_file):
        print('Loading stats from {} files'.format(train_stats_file))
        train_stats = np.load(train_stats_file)
        class_means = train_stats['class_means']
        class_vars  = train_stats['class_vars']
    else:
        if 'deeplabv3plus_r101-d8' in dir:
            print('Loading hardcoded stats for DLv3+ R101')
            '''class_means = np.array([9.245036 , 9.03187  , 9.081645 , 7.6389194, 7.596713 , 8.109738 ,
                7.4709554, 8.309173 , 9.198976 , 7.7139444, 8.799796 , 8.097538 ,
                7.2400327, 9.727406 , 7.582097 , 6.83076  , 6.517006 , 6.925336 ,
                7.2537894])
            class_vars = np.array([0.20769611, 0.9072598 , 0.7005657 , 0.97401094, 0.98806524,
                1.3872538 , 0.8447538 , 1.0696217 , 0.99043494, 0.8490232 ,
                0.35153958, 1.3806854 , 1.3625157 , 2.5833855 , 1.0239671 ,
                0.88884085, 0.8729633 , 1.0973964 , 1.2293454 ])'''
            class_means = np.array([9.08941269, 7.89703083, 8.17335129, 6.33317661, 6.43940592,
                    6.93818569, 6.31334782, 6.80468702, 8.14266872, 6.85002375,
                    7.37686396, 6.92912626, 5.62570429, 8.23272228, 5.52825022,
                    5.47206068, 5.05167723, 5.15879631, 6.20896339])
            class_vars = np.array([2.55288291, 1.81396711, 2.16587615, 1.89423168, 1.78679729,
                    2.07765007, 1.50867808, 2.06212974, 1.90717423, 1.70643413,
                    1.74401367, 2.09695888, 2.1290369 , 1.9848628 , 2.45909405,
                    2.29201245, 1.55197382, 2.02900386, 1.87817681])
        elif 'segformer_b2' in dir:
            print('Loading hardcoded stats for SegFormer-B2')
            class_means = np.array([5.3805456, 3.6617112, 2.3302677, 1.9280645, 2.1789162, 2.9806347,
                2.743042 , 3.3491263, 3.8155932, 2.8279028, 4.810911 , 3.0563781,
                2.7660158, 3.814196 , 1.9745747, 1.6802019, 2.0725608, 2.4728982,
                2.6950345])
            class_vars = np.array([2.1149895 , 1.3118162 , 0.59761196, 0.8517033 , 1.024318  ,
                1.8134747 , 1.0817857 , 1.9584851 , 1.0542632 , 1.1857089 ,
                1.0924104 , 1.9609584 , 1.6634685 , 1.5961114 , 1.0449332 ,
                0.9658174 , 0.84914696, 1.2382355 , 1.3940778 ])
        elif 'segformer_b5' in dir:
            print('Loading hardcoded stats for SegFormer-B5')
            class_means = np.array([5.5343714, 3.6296852, 2.6308858, 2.2637699, 2.7084043, 3.579329 ,
                3.196027 , 4.2705283, 4.094109 , 3.1227248, 4.4199615, 3.9328082,
                3.1804786, 4.207104 , 2.4737117, 2.8840616, 3.03795  , 2.9464016,
                3.1708856])
            class_vars = np.array([2.6823127 , 1.3747933 , 0.93583477, 0.9714503 , 1.4102285 ,
                2.3407507 , 1.3723124 , 3.440775  , 1.3492926 , 1.3778759 ,
                1.0351747 , 2.6922424 , 2.266627  , 2.3636127 , 1.2852088 ,
                1.2875468 , 1.2889034 , 1.4584093 , 1.9024252])
        else:
            raise NotImplementedError('{} does not have hard-coded stats!'.format(dir))
    
    return class_means, class_vars
