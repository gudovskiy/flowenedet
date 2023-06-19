from abc import ABCMeta, abstractmethod
import numpy as np
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy, auroc
from ..utils import save_stats, load_stats, gf_kernel, BoundarySuppressionWithSmoothing
from flows.all_flows import FlowMixDet

actLogSigm = nn.LogSigmoid()
actSoft    = nn.Softplus()

#self.flow_model = build_flow(ood_type, coupling_blocks, num_mixtures, conditions, channels, dropout_ratio)
def build_flow(ood_type, coupling_blocks, channels, conditions, latents, dropout=0.0):
    cond = True if 'cflow' in ood_type else False
    if '-cflow' in ood_type or '-flow' in ood_type:
        L, C, P, Z = coupling_blocks, channels, conditions, latents
        G = C % 2  # number of extra augmentation dimensions (to make divisible by 2 e.g. for Cityscapes)
        D = C + G  # total number of data dimensions
        #act = 'relu'  # {'softplus', 'relu', 'elu', 'gelu', 'tanh', 'celu', 'crelu', 'sigmoid'}
        act = 'sigmoid'
        #act = 'selu'
        #act = 'tanh'
        dst = 'MGM' # 'GM'
        ltype = 'conv2d'  # {'conv2d', 'wnconv2d', 'snconv2d'}
        return FlowMixDet(L, D, P, Z, ltype=ltype, act=act, cond=cond, dst=dst, dropout=dropout)
    elif '-lflow' in ood_type:
        L, C, P, Z = coupling_blocks, conditions, 0, channels
        G = C % 2  # number of extra augmentation dimensions (to make divisible by 2 e.g. for Cityscapes)
        D = C + G  # total number of data dimensions
        act = 'relu'  # {'softplus', 'relu', 'elu', 'gelu', 'tanh', 'celu', 'crelu'}
        dst = 'LGM'  # 'GM'
        ltype = 'conv2d'  # {'conv2d', 'wnconv2d', 'snconv2d'}
        return FlowMixDet(L, D, P, Z, ltype=ltype, act=act, cond=cond, dst=dst, dropout=dropout)
    else:
        raise NotImplementedError('{} is not supported FLOW type!'.format(ood_type))


class FlowDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for FlowDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                in_channels,
                channels,
                *,
                num_classes,
                out_channels=None,
                threshold=None,
                dropout_ratio=0.1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                in_index=-1,
                input_transform=None,
                flow_decode=dict(model='LOGITFLOWFMD', type='realnvp-cflow', coupling_blocks=2, conditions=32),
                loss_decode=dict(model='LOGITFLOWFMD', type='FlowLoss', num_classes=-1),
                ignore_index=255,
                sampler=None,
                align_corners=False,
                init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                # new params:
                ood_class_index=19,
                ood_est_stats=False,
                ood_dir_stats=None,
                freeze=False,
                flow_upsample=False,
                post_processing=False
        ):
        super(FlowDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        ######## OOD model ########
        self.freeze = freeze
        self.ood_class_index = ood_class_index

        if self.freeze:
            print('Linear classifier is frozen!')
            self.conv_seg.eval()

        ood_model = flow_decode['model']
        self.ood_model = ood_model
        if   'FMD' in ood_model:
            num_mixtures = 2
        else:
            raise NotImplementedError('{} is not supported OOD model!'.format(ood_model))
        # loss
        class_weight = num_mixtures*[1.0]  # for training
        loss_decode['class_weight'] = class_weight
        
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.loss_decode = build_loss(loss_decode)
        self.num_mixtures = num_mixtures
        # OOD type
        ood_type = flow_decode['type']
        self.ood_type = ood_type
        coupling_blocks = flow_decode['coupling_blocks']
        conditions = flow_decode['conditions']
        self.conditions = conditions
        self.flow_model = build_flow(ood_type, coupling_blocks, num_mixtures, conditions, channels, dropout_ratio)
        
        self.flow_upsample = flow_upsample
        self.post_processing = post_processing
        if self.post_processing:  # SML post-processing:
            self.post_processor = BoundarySuppressionWithSmoothing(boundary_suppression=True, boundary_width=4, boundary_iteration=4, dilated_smoothing=True, kernel_size=7, dilation_size=6)
        else:
            self.post_processor = gf_kernel(7, 1, sigma=1)  # Gaussian filter
        # projection layer:
        if '-cflow' in ood_type or '-lflow' in ood_type:
            self.proj_seg = nn.AvgPool1d(channels//conditions)
        else:
            self.proj_seg = nn.Identity()
        
        self.ood_est_stats = ood_est_stats
        self.ood_dir_stats = ood_dir_stats
        if ood_est_stats:
            class_means = torch.zeros(num_classes)
            class_vars  = torch.ones(num_classes)
        else:
            class_means, class_vars = load_stats(ood_dir_stats)
            class_means = torch.from_numpy(class_means).float()
            class_vars  = torch.from_numpy(class_vars).float()
        #
        self.class_means = nn.Parameter(class_means, requires_grad=False)
        self.class_vars  = nn.Parameter(class_vars,  requires_grad=False)

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        with torch.no_grad():
            linear_logits, feats = self.forward(inputs)

        # train flow only
        flow_logits, flow_dists = self.flow_seg(feats, linear_logits)
        losses = self.losses(linear_logits, flow_logits, flow_dists, gt_semantic_seg)

        if self.ood_est_stats:  # estimate logit statistics
            stat_labels = gt_semantic_seg
            stat_logits = resize(input=linear_logits, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=self.align_corners)
            save_stats(self.class_means, self.class_vars, stat_labels, stat_logits, self.ood_dir_stats, self.num_classes, self.ignore_index)
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        linear_logits, feats = self.forward(inputs)
        B, _, H, W = linear_logits.shape
        S = H * W
        if 'crop_size' in test_cfg:
            inpSize = test_cfg['crop_size']  # 4x of the logit shape
        else:
            inpSize = img_metas[0]['ori_shape'][:2]  # 4x of the logit shape
        
        feats = self.proj_seg(feats.transpose(1,3).reshape(B,S,-1)).reshape(B,W,H,-1).transpose(1,3)

        if self.flow_upsample:  # upsampling inputs to flow model
            resize_linear_logits = resize(input=linear_logits, size=inpSize, mode='bilinear', align_corners=self.align_corners)
            resize_feats         = resize(input=feats,         size=inpSize, mode='bilinear', align_corners=self.align_corners)
        else:
            resize_linear_logits = linear_logits
            resize_feats = feats

        _, flow_dists = self.flow_seg(resize_feats, resize_linear_logits, test=True)
        
        if self.post_processing:
            pp_linear_logits = resize(input=resize_linear_logits, size=inpSize, mode='bilinear', align_corners=self.align_corners)
            pp_flow_dists    = resize(input=flow_dists,           size=inpSize, mode='bilinear', align_corners=self.align_corners)
            _, pp_linear_argmax  = torch.max(pp_linear_logits, dim=1, keepdim=True)  # argmax
            return [linear_logits, self.post_processor(pp_flow_dists, pp_linear_argmax)]
        else:
            return [linear_logits, self.post_processor(flow_dists)]

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def flow_seg(self, feats, linear_logits, test=False):
        """Flow-classify each pixel."""
        feats, linear_logits = feats.detach(), linear_logits.clone().detach()
        device = linear_logits.device
        B, _, H, W = linear_logits.shape
        S = H * W
        E = B * S
        M  = self.num_mixtures
        CL = self.num_classes
        logprobs = linear_logits.clone()
        if not self.ood_est_stats:
            logprobs -=            self.class_means.view(1,-1,1,1)
            logprobs /= torch.sqrt(self.class_vars.view( 1,-1,1,1))
        
        margs = -actSoft(-torch.logsumexp(logprobs, dim=1, keepdim=True))  # log P(in|x)
        nargs = torch.log(1e-6+1.0 - torch.exp(margs))  # log P(out|x)
        #margs = torch.sigmoid(torch.logsumexp(logprobs, dim=1, keepdim=True))  # log P(in|x)
        #nargs = 1 - margs

        if 'FMD' in self.ood_model:
            z = torch.cat((margs, nargs), dim=1)
        
        _, C, _, _ = z.shape

        if '-flow' in self.ood_type:
            logZYM = self.flow_model.log_prob(z, context=None)  # ExDenseM+Background: log(z|y=m)
        elif '-cflow' in self.ood_type:
            if test:
                context = feats
            else:
                context = self.proj_seg(feats.transpose(1,3).reshape(B,S,-1)).reshape(B,W,H,-1).transpose(1,3)

            logZYM = self.flow_model.log_prob(z, context=context)  # ExDenseM+Background: log(z|y=m)
        #elif '-lflow' in self.ood_type:
        #    #context = self.proj_seg(feats.transpose(1,3).reshape(B,S,-1)).reshape(B,W,H,-1).transpose(1,3)
        #    context = self.proj_seg(feats)
        #    logZYM = self.flow_model.log_prob(p, context=context)  # ExDenseM+Background: log(z|y=m)
        else:
            raise NotImplementedError('{} is not supported OOD type!'.format(self.ood_type))
        #
        pYMZ = F.softmax(logZYM.clone().detach(), dim=1)
        if 'FMD' in self.ood_model:
            logP = pYMZ[:,1,...].unsqueeze(1)
        
        logits = logZYM
        dists  = logP
        return logits, dists

    @force_fp32(apply_to=('linear_logit', 'flow_logit', 'flow_dist'))
    def losses(self, linear_logit, flow_logit, flow_dist, label):
        """Compute segmentation loss."""
        loss = dict()

        linear_logit = resize(input=linear_logit, size=label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        flow_logit   = resize(input=flow_logit,   size=label.shape[2:], mode='bilinear', align_corners=self.align_corners)

        CL = self.num_classes
        ood_model = self.ood_model
        label = label.squeeze(1)
        if 'FMD' in ood_model:
            ignore_index = -100
            argSoft = torch.argmax(linear_logit, dim=1)
            label_mix = (label != argSoft).long()

        if self.sampler is not None:
            weight = self.sampler.sample(flow_logit, label_mix.unsqueeze(1))
        else:
            weight = None

        losses = self.loss_decode(flow_logit, label_mix, weight=weight, ignore_index=ignore_index)
        loss['closs'] = losses[0]
        loss['aloss'] = losses[1]

        #loss['acc'] = accuracy(flow_logit, label)
        #loss['ood'] = auroc(linear_logit, flow_dist, label, self.ood_class_index)
        return loss
