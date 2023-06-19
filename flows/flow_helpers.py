import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def checkerboard(S, K):
    """
        shape: dimensions of output tensor
        k: edge size of square
    """
    H, W = S
    indices = torch.stack(torch.meshgrid(torch.arange(H//K), torch.arange(W//K), indexing='ij'))
    base = indices.sum(dim=0) % 2
    x = base.repeat_interleave(K, 0).repeat_interleave(K, 1)
    return 1-x


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        #multi_inp = False
        #if len(input) > 1:
        #    multi_inp = True
        #    _, edge_index = input[0], input[1]

        for module in self._modules.values():
            #if multi_inp:
            #    if hasattr(module, 'weight'):
            #        input = [module(*input)]
            #    else:
            #        # Only pass in the features to the Non-linearity
            #        input = [module(input[0]), edge_index]
            #else:
            input = [module(*input)]
        return input[0]

# --------------------
# Model layers and helpers
# --------------------

class CELU(nn.Module):
    """ ConcatELU layer """
    def __init__(self):
        super().__init__()
        self.act = nn.ELU(inplace=False)

    def forward(self, x):
        return self.act(torch.cat((x, -x), dim=1))


class CRELU(nn.Module):
    """ ConcatRELU layer """
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(torch.cat((x, -x), dim=1))


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, padding_mode='reflect'):
        super(WNConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        if self.training:
            self.conv = nn.utils.weight_norm(conv)
        else:
            self.conv = nn.utils.remove_weight_norm(conv)

    def forward(self, x):
        return self.conv(x)


class SNConv2d(nn.Module):
    """Weight-normalized 2d convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, padding_mode='reflect'):
        super(SNConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        if self.training:
            self.conv = nn.utils.spectral_norm(conv)
        else:
            self.conv = nn.utils.remove_spectral_norm(conv)

    def forward(self, x):
        return self.conv(x)


kwargs_layer = {'conv2d': nn.Conv2d, 'wnconv2d': WNConv2d, 'snconv2d': SNConv2d}  #, 'GCN': GCNConv, 'GAT': GATConv}
kwargs_act   = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'elu': nn.ELU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'selu': nn.SELU(),
                                            'crelu': CRELU(),  'celu': CELU(), 'sigmoid': nn.Sigmoid()}


class RealNVP(nn.Module):
    """ Real NVP coupling (https://arxiv.org/abs/1605.08803)
    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, D, P, M, K=0, ltype='conv2d', act='relu', cond=False, dropout=0.0):
        super(RealNVP, self).__init__()
        self.D, self.P, self.K  = D, P, K
        A = cond*P+D
        ACT= kwargs_act[act]
        C1 = kwargs_layer[ltype](A, A, 1)
        #C2 = kwargs_layer[ltype](A, A, 3, padding=1, padding_mode='reflect')
        C2 = kwargs_layer[ltype](A, A, 7, padding=3, padding_mode='reflect')
        #C2 = kwargs_layer[ltype](A, A, 11, padding=5, padding_mode='reflect')
        C3 = kwargs_layer[ltype](A, D, 1)
        if act == 'selu':  # to get a proper conv gain
            torch.nn.init.kaiming_uniform_(C1.weight, mode='fan_in', nonlinearity='linear')
            torch.nn.init.kaiming_uniform_(C2.weight, mode='fan_in', nonlinearity='linear')
            torch.nn.init.kaiming_uniform_(C3.weight, mode='fan_in', nonlinearity='linear')
        # checkerboard
        if K > 0:  # spatial&channel-wise checkerboard
            self.M = nn.Parameter(M.view(1, D, K, K), requires_grad=False)
        else:  # channel-wise checkerboard
            self.M = nn.Parameter(M.view(1, D, 1, 1), requires_grad=False)
        # dropout:
        pDrop = 0.2  # dropout
        if pDrop > 0.0:
            DROP = nn.Dropout2d(pDrop)  # nn.AlphaDropout(pDrop)
        else:
            DROP = nn.Identity()
        
        self.s = nn.Sequential(C1, ACT, C2, ACT, DROP, C3)
        self.t = nn.Sequential(C1, ACT, C2, ACT, DROP, C3)

        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

    def forward(self, x, context=None):
        B, D, H, W = x.shape
        if self.K > 0:  # spatial&channel-wise checkerboard
            M = torch.tile(self.M, (1, 1, H//self.K+1, W//self.K+1))[...,:H,:W]
        else:
            M = self.M
        xM = x*M
        # concat
        if context is not None:
            xC = torch.cat((xM, context), dim=1)
        else:
            xC = x
        # coupling
        sout = self.s(xC)
        logs =  - self.actSoft(sout)  # (-Inf:0)
        s = 1.0 - self.actSigm(sout)  # (0:1)
        t = self.t(xC)
        # output
        x   = xM + (1 - M) * (x*s + t)
        ldj = (1 - M) * logs  # log det dx/du
        return x, ldj

    def inverse(self, x, context=None):
        B, D, H, W = x.shape
        if self.K > 0:  # spatial&channel-wise checkerboard
            M = torch.tile(self.M, (1, 1, H//self.K+1, W//self.K+1))[...,:H,:W]
        else:
            M = self.M
        xM = x*M
        # concat
        if context is not None:
            xC = torch.cat((xM, context), dim=1)
        else:
            xC = x
        # coupling
        sout = self.s(xC)
        logs =  - self.actSoft(sout)  # (-Inf:0)
        s = 1.0 - self.actSigm(sout)  # (0:1)
        t = self.t(xC)
        # output
        x   = (1 - M) * (x - t) / s
        ldj = (1 - M) * logs
        return x, ldj


def create_rnvp_blocks(D, P, L, M, K=0, ltype='conv2d', act='relu', cond=False, dropout=0.0):
    net = []
    for i in range(L):
        net += [RealNVP(D, P, M[i], K=K, ltype=ltype, act=act, cond=cond, dropout=dropout)]
    return MultiInputSequential(*net)


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, D, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.dim = (0, 2, 3)
        dim_param = (1, D, 1, 1)

        self.log_gamma = nn.Parameter(torch.zeros(*dim_param))
        self.beta = nn.Parameter(torch.zeros(*dim_param))
        self.register_buffer('running_mean', torch.zeros(*dim_param))
        self.register_buffer('running_var', torch.ones(*dim_param))

        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

    def forward(self, x):
        if self.training:
            self.batch_mean = torch.mean(x, self.dim, keepdim=True)
            self.batch_var = torch.var(x, self.dim, keepdim=True)  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var  = self.batch_var
        else:
            mean = self.running_mean
            var  = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        log_gamma = - self.actSoft(self.log_gamma)
        gamma = 1.0 - self.actSigm(self.log_gamma)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = gamma * x_hat + self.beta

        # compute ldj (cf RealNVP paper)
        ldj = log_gamma - 0.5 * torch.log(var + self.eps)
        # print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        # (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), ldj.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, ldj

    def inverse(self, y):
        if self.training:
            mean = torch.mean(y, self.dim, keepdim=True)
            var  = torch.var(y, self.dim, keepdim=True)
        else:
            mean = self.running_mean
            var  = self.running_var

        log_gamma = - self.actSoft(self.log_gamma)
        gamma = 1.0 - self.actSigm(self.log_gamma)
        x_hat = (y - self.beta) / gamma
        x = x_hat * torch.sqrt(var + self.eps) + mean

        ldj = 0.5 * torch.log(var + self.eps) - log_gamma

        return x, ldj


class ActNorm(nn.Module):
    """ Glow ActNorm layer """
    def __init__(self, D, data_dep_init=True, eps=1e-6):
        super().__init__()
        self.eps = eps
    
        self.dim = (0, 2, 3)
        dim_param = (1, D, 1, 1)

        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.shift     = nn.Parameter(torch.zeros(*dim_param))
        self.log_scale = nn.Parameter(torch.zeros(*dim_param))
        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

    def data_init(self, x):
        self.initialized += 1.0
        with torch.no_grad():
            x_mean = torch.mean(x, self.dim, keepdim=True)
            x_std  = torch.std( x, self.dim, keepdim=True)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def forward(self, x):
        if self.training and not self.initialized: self.data_init(x)
        # compute normalized input (cf original batch norm paper algo 1)
        log_scale = - self.actSoft(self.log_scale)
        scale = 1.0 - self.actSigm(self.log_scale)
        y = self.shift + x * scale
        ldj = log_scale
        return y, ldj

    def inverse(self, y):
        log_scale = - self.actSoft(self.log_scale)
        scale = 1.0 - self.actSigm(self.log_scale)
        x = (y - self.shift) / scale
        ldj = log_scale
        return x, ldj


class InstanceNorm(nn.Module):
    """ RealNVP InstanceNorm layer """
    def __init__(self, D, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
    
        self.dim = (2, 3)
        dim_param = (1, D, 1, 1)

        self.log_gamma = nn.Parameter(torch.zeros(*dim_param))
        self.beta = nn.Parameter(torch.zeros(*dim_param))

        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

    def forward(self, x):
        mean = torch.mean(x, self.dim, keepdim=True)
        var  = torch.var( x, self.dim, keepdim=True)
        # compute normalized input (cf original batch norm paper algo 1)
        log_gamma = - self.actSoft(self.log_gamma)
        gamma = 1.0 - self.actSigm(self.log_gamma)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = gamma * x_hat + self.beta
        # compute ldj (cf RealNVP paper)
        ldj = log_gamma - 0.5 * torch.log(var + self.eps)
        return y, ldj

    def inverse(self, y):
        mean = torch.mean(y, self.dim, keepdim=True)
        var  = torch.var(y, self.dim, keepdim=True)
        # compute normalized input (cf original batch norm paper algo 1)
        log_gamma = - self.actSoft(self.log_gamma)
        gamma = 1.0 - self.actSigm(self.log_gamma)
        x_hat = (y - self.beta) / gamma
        x = x_hat * torch.sqrt(var + self.eps) + mean
        # compute ldj (cf RealNVP paper)
        ldj = 0.5 * torch.log(var + self.eps) - log_gamma
        return x, ldj


class Shuffle(nn.Module):
    """ Shuffle layer """
    def __init__(self, D):
        super().__init__()
        self.register_buffer('idx', torch.randperm(D))

    def forward(self, x):
        y = torch.index_select(x, 1, self.idx)
        return y

    def inverse(self, y):
        x = torch.index_select(y, 1, torch.argsort(self.idx))
        return y


kwargs_norm = {'bn': BatchNorm, 'an': ActNorm, 'in': InstanceNorm}
def create_norm_blocks(D, L, norm='bn'):
    net = []
    for i in range(L):
        net += [kwargs_norm[norm](D)]
    return MultiInputSequential(*net)

def create_perm_blocks(D, L):
    net = []
    for i in range(L):
        net += [Shuffle(D)]
    return MultiInputSequential(*net)


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
    Args:
        num_channels (int): Number of channels in the input and output.
        random_init (bool): Initialize with a random orthogonal matrix.
            Otherwise initialize with noisy identity.
    """
    def __init__(self, D, random_init=False):
        super(InvConv, self).__init__()
        self.D = D
        self.weight = nn.Parameter(torch.Tensor(D, D))
        if random_init:
            bound = 1.0 / np.sqrt(D)
            nn.init.uniform_(D, -bound, bound)
        else:
            nn.init.orthogonal_(self.weight)

    def forward(self, x):
        ldj = torch.slogdet(self.weight)[1]
        weight = self.weight.view(self.D, self.D, 1, 1)
        x = F.conv2d(x, weight)
        return x, ldj

    def inverse(self, x):
        ldj = torch.slogdet(self.weight)[1]
        weight = torch.inverse(self.weight.double()).float()
        weight = self.weight.view(self.D, self.D, 1, 1)
        x = F.conv2d(x, weight)
        return x, ldj


def create_invconv_blocks(D, L):
    net = []
    for i in range(L):
        net += [InvConv(D)]
    return MultiInputSequential(*net)


class IMap(nn.Module):
    """Generative Flows with Invertible Attentions (https://arxiv.org/pdf/2106.03959.pdf)
    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, D, P, M, K, ltype='conv2d', act='relu', cond=False):
        super(IMap, self).__init__()
        self.D, self.P, self.K  = D, 0, K
        weight = torch.empty(D, D, 1)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)
        bias = torch.empty(D)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.bias = nn.Parameter(bias)
        self.register_parameter('s', nn.Parameter(torch.randn(1, D, 1)))
        #self.register_parameter('offset', nn.Parameter(torch.ones(1, 1, S)))  # Bx1xS=HW self.offset is WRONG (not in the paper)?
        self.act = nn.Sigmoid()
        self.actSoft = nn.Softplus()  # [0:+Inf]
        self.actSigm = nn.Sigmoid()  # (0:1)

        # checkerboard
        if K > 0:  # spatial&channel-wise checkerboard
            self.M = nn.Parameter(M.view(1, D, K, K), requires_grad=False)
        else:  # channel-wise checkerboard
            self.M = nn.Parameter(M.view(1, D, 1), requires_grad=False)

    def forward(self, x):
        B, D, H, W = x.shape
        if self.K > 0:  # spatial&channel-wise checkerboard
            M = torch.tile(self.M, (1, 1, H//self.K+1, W//self.K+1))[...,:H,:W].reshape(1,D,-1)
        else:
            M = self.M
        xM = x.view(B,D,-1)*M
        z = F.conv1d(xM, self.weight, bias=self.bias)  # BxDxHW
        pool_out = torch.mean(z, dim=1, keepdim=True)  # Bx1xHW
        log_attn = - self.actSoft(pool_out)  # (-Inf:0)
        attn = 1.0 - self.actSigm(pool_out)  # (0:1)
        log_lvar = - self.actSoft(self.s)  # (-Inf:0)
        lvar = 1.0 - self.actSigm(self.s)  # (0:1)
        attn_mask = (1-M) * attn + M * lvar
        y = x * attn_mask.view(B, D, H, W)
        ldj0 = D/2 * log_attn
        ldj1 = log_lvar * M
        ldj = ldj0 + ldj1
        return y, ldj.view(B, D, H, W)

    def inverse(self, x):
        B, D, H, W = x.shape
        if self.K > 0:  # spatial&channel-wise checkerboard
            M = torch.tile(self.M, (1, 1, H//self.K+1, W//self.K+1))[...,:H,:W].reshape(1,D,-1)
        else:
            M = self.M
        log_lvar = - self.actSoft(self.s)  # (-Inf:0)
        lvar = 1.0 - self.actSigm(self.s)  # (0:1)
        lvar = torch.ones_like(lvar) / lvar
        xM = x.view(B,D,-1)*M
        xM = M * x.view(B,D,-1) * lvar
        z = F.conv1d(xM, self.weight, bias=self.bias)  # BxDxHW
        pool_out = torch.mean(z, dim=1, keepdim=True)  # Bx1xHW
        log_attn = - self.actSoft(pool_out)  # (-Inf:0)
        attn = 1.0 - self.actSigm(pool_out)  # (0:1)
        attn =  torch.ones_like(attn) / attn
        attn_mask = (1-M) * attn + M * lvar
        y = x * attn_mask.view(B, D, H, W)
        ldj0 = D/2 * log_attn
        ldj1 = log_lvar * M
        ldj = ldj0 + ldj1
        return y, ldj.view(B, D, H, W)


def create_imap_blocks(D, P, L, M, K=0, ltype='conv2d', act='relu', cond=False):
    net = []
    for i in range(L):
        net += [IMap(D, P, M[i], K=K, ltype=ltype, act=act, cond=cond)]
    return MultiInputSequential(*net)
