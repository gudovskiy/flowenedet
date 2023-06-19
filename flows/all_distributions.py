import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

INVSOFTPLUS = 1.4427
LOGPI       = 1.8378770664093453

class GaussianMixture(nn.Module):
    """A Normal with mean and diagonal covariance matrix"""
    def __init__(self, C, ltype='linear'):
        super().__init__()
        self.C = C
        self.actSoft = nn.Softplus()  # [0:+Inf)

        if ltype == 'linear':
            dim_w = (1, C)
            dim_s = (   C)
        else:
            dim_w = (1, C, 1, 1)
            dim_s = (   C, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim_w), requires_grad=True)  # 1xC
        self.loc    = nn.Parameter(torch.zeros(dim_s), requires_grad=True)  # C
        self.scale  = nn.Parameter(torch.zeros(dim_s), requires_grad=True)  # C

    def cond_dist(self, context):
        loc = self.loc  # (-Inf:+Inf), init = 0
        scale = INVSOFTPLUS*self.actSoft(self.scale)  # [0:+Inf), init = 1
        return Normal(loc, scale=scale)

    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None):
        dist = self.cond_dist(context)
        logW = -self.actSoft(self.weight)
        logP = dist.log_prob(input)
        log_probs = logW + logP  # BxC
        return log_probs
    
    def sample(self, n_samples, context=None):
        dist = self.cond_dist(context)
        x = dist.rsample((n_samples,))
        x_context = x[torch.arange(n_samples), context].view(n_samples, *self.size)
        log_px = self.log_prob(x_context, context)
        return x_context, log_px


class MultivariateGaussianMixture(nn.Module):
    """A Normal with mean and full covariance matrix"""
    def __init__(self, C, ltype='linear'):
        super().__init__()
        self.C = C
        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

        dim_w = (C, 1)
        dim_s = (C, 1)
        dim_t = (C*(C+1)//2)

        self.w  = nn.Parameter(torch.zeros(dim_w), requires_grad=True)  # Cx1
        self.mu = nn.Parameter(torch.zeros(dim_s), requires_grad=True)  # Cx1
        self.U  = nn.Parameter(torch.zeros(dim_t), requires_grad=True)  # U
        self.idxT = torch.tril_indices(C, C, offset=-1)

    def forward(self, x, context=None, rev=False):
        return self.log_prob(x, context, rev)

    def log_prob(self, x, context=None, rev=False):
        # Get GMM parameters
        B, _, H, W = x.shape
        C = self.C
        w, mu, U_entries, idxT = self.w, self.mu, self.U, self.idxT
        logW = -self.actSoft(w)
        # Construct upper triangular Cholesky factors U of all precision matrices
        #logD =     - self.actSoft(U_entries[:C]).unsqueeze(-1)  # log-diagonal (-Inf:0)
        #D    = 1.0 - self.actSigm(U_entries[:C])  # diagonal (0:+Inf)
        D = INVSOFTPLUS*self.actSoft(U_entries[:C])  # [0:+Inf), init = 1
        logD = torch.log(D).unsqueeze(-1)
        T = U_entries[C:]  # tril (-Inf:+Inf)
        U = torch.diag(D)  # log diag of U (E^{-1} = U^{T} U)
        U[idxT[0], idxT[1]] = T
        x = x.transpose(0,1).reshape(C,-1)  # CxB*H*W
        if rev:
            logD*= -1.0
            logP = -0.5 * (torch.matmul(torch.inverse(U), (x + mu)))**2
        else:
            logP = -0.5 * (torch.matmul(              U,  (x - mu)))**2
        
        log_probs = logW + logD + logP  # CxB*H*W - nll_upper_bound
        log_probs = log_probs.reshape(C,B,H,W).transpose(0,1)  # BxCxHxW

        return log_probs


class LatentGaussianMixture(nn.Module):
    """A Normal with mean and diagonal covariance matrix"""
    def __init__(self, M, C, ltype='linear'):
        super().__init__()
        self.C = C
        self.actSoft = nn.Softplus()  # [0:+Inf)

        dim_w = (1, M, 1, 1)
        dim_s = (M, C, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim_w), requires_grad=True)
        self.loc    = nn.Parameter(torch.zeros(dim_s), requires_grad=True)
        self.scale  = nn.Parameter(torch.zeros(dim_s), requires_grad=True)

    def cond_dist(self, context):
        loc = self.loc  # (-Inf:+Inf), init = 0
        scale = INVSOFTPLUS*self.actSoft(self.scale)  # [0:+Inf), init = 1
        return Normal(loc, scale=scale)

    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None):
        dist = self.cond_dist(context)
        weight = INVSOFTPLUS*self.actSoft(self.weight)
        logP = dist.log_prob(input.unsqueeze(1))  # BxMxCxHxW
        log_probs = F.log_softmax(weight, dim=1) + torch.sum(logP, dim=2)  # BxMxHxW
        return log_probs
    
    def sample(self, n_samples, context=None):
        dist = self.cond_dist(context)
        x = dist.rsample((n_samples,))
        x_context = x[torch.arange(n_samples), context].view(n_samples, *self.size)
        log_px = self.log_prob(x_context, context)
        return x_context, log_px
