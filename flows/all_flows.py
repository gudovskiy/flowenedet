import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flows.flow_helpers import *
from flows.all_distributions import GaussianMixture, MultivariateGaussianMixture, LatentGaussianMixture

# SurVAE-type scale functions:
kwargs_sc = {'exp': lambda s: torch.exp(s), 'softplus': lambda s: F.softplus(s), \
             'sigmoid': lambda s: torch.sigmoid(s + 2.) + 1e-3, 'tanh_exp': lambda s: torch.exp(2.*torch.tanh(s/2.))}

# available base distributions
kwargs_dsts = {'GM': GaussianMixture, 'MGM': MultivariateGaussianMixture, 'LGM': LatentGaussianMixture}

# --------------------
# Models
# --------------------

## Taken from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class FlowMixDet(nn.Module):
    def __init__(self, L, D, P, Z, ltype='conv2d', act='relu', cond=False, dst='GM', dropout=0.0):
        super(FlowMixDet, self).__init__()
        self.L = L  # num of coupling blocks
        self.D = D  # input/hidden dim
        self.dst = dst
        # checkerboard
        fM = torch.arange(D).float() % 2
        # spatial & channel-wise checkerboard
        K = 0  # channel-wise mask
        #K = 8  # 4x4 channel+HW mask
        if K>0:
            fS = checkerboard((K,K), K//2)
            fSM = torch.matmul(  fM.view(-1,1), 1.0 * fS.view(1,-1)) \
                + torch.matmul(1-fM.view(-1,1), 1.0 - fS.view(1,-1))
            fSM = fSM.view(-1,K,K)
            M = torch.stack([fSM, 1-fSM]).repeat(L//2, 1, 1, 1)
        else:
            M = torch.stack([fM, 1-fM]).repeat(L//2, 1)
        # base distribution:
        if dst == 'LGM':
            self.p_z = kwargs_dsts[dst](Z, D, ltype)
        else:
            self.p_z = kwargs_dsts[dst](D, ltype)
        # coupling block:
        #self.proj = create_proj_blocks(Z, P, L)
        self.rnvp = create_rnvp_blocks(D, P, L, M, K=K, ltype=ltype, act=act, cond=cond, dropout=dropout)
        self.cn   = create_norm_blocks(P, L, norm='an')
        self.an   = create_norm_blocks(D, L, norm='an')
        self.ic   = create_invconv_blocks(D, L)  # self.ic = L*[None] # W^{-1} like in Glow paper
        self.sh   = L*[None]  # self.sh = create_perm_blocks(D, L)
        self.imap = create_imap_blocks(D, P, L, M, K=K, ltype=ltype, act=act, cond=cond)
        #self.imap = L*[None]

    def forward(self, x, context=None):
        sldj, x = torch.zeros_like(x), x
        #
        for i in range(self.L):
            # condition
            if context is not None and self.cn[i] is not None:
                p = context  # p = self.proj[i](context)
                c, _ = self.cn[i].forward(p)
            else:
                c = None
            # normalization
            if self.an[i] is not None:
                x, ldj = self.an[i].forward(x)
                sldj += ldj
            # W^{-1}
            if self.ic[i] is not None:
                x, ldj = self.ic[i].forward(x)
                sldj += ldj
            # Real NVP
            x, ldj = self.rnvp[i].forward(x, c)
            sldj += ldj
            # iMap
            if self.imap[i] is not None:
                x, ldj = self.imap[i].forward(x)
                sldj += ldj
            #
        return x, sldj

    def inverse(self, x, context=None):
        sldj, x = torch.zeros_like(x), x
        #
        for i in reversed(range(self.L)):
            # condition
            if context is not None and self.cn[i] is not None:
                p = context  # p = self.proj[i](context)
                c, _ = self.cn[i].inverse(p)
            else:
                c = None
            # normalization
            if self.an[i] is not None:
                x, ldj = self.an[i].inverse(x)
                sldj -= ldj
            # W^{-1}
            if self.ic[i] is not None:
                x, ldj = self.ic[i].inverse(x)
                sldj -= ldj
            # Real NVP
            x, ldj = self.rnvp[i].inverse(x, c)
            sldj -= ldj
            # iMap
            if self.imap[i] is not None:
                x, ldj = self.imap[i].inverse(x)
                sldj -= ldj
            #
        return x, sldj

    def log_prob(self, x, context=None):
        z, ldj = self.forward(x, context)  # BxCxHxW
        logp = self.p_z(z)  # BxMxHxW
        if self.dst == 'LGM':
            return torch.sum(ldj, dim=1, keepdim=True) + logp
        else:
            return ldj + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x
