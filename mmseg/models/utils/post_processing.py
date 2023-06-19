import torch, math
import torch.nn as nn
import numpy as np
from scipy import ndimage as ndi
from kornia.morphology import dilation, erosion

def gf_kernel(kernel, channels, sigma=1):
    # Set these to whatever you want for your gaussian filter
    K = kernel
    C = channels
    P = K//2
    sigma = sigma
    # Create a x, y coordinate grid of shape (K, K, 2)
    x_cord = torch.arange(K)
    x_grid = x_cord.repeat(K).view(K, K)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    #
    mean = (K - 1)/2.
    variance = sigma**2.
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, K, K)
    gaussian_kernel = gaussian_kernel.repeat(C, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(C, C, K, padding=P, groups=C, bias=False, padding_mode='reflect')
    gaussian_filter.weight.data = torch.Tensor(gaussian_kernel)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class BoundarySuppressionWithSmoothing(nn.Module):
    """
    Apply boundary suppression and dilated smoothing
    """
    def __init__(self, boundary_suppression=True, boundary_width=4, boundary_iteration=4, dilated_smoothing=True, kernel_size=7, dilation_size=6):
        super(BoundarySuppressionWithSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.boundary_suppression = boundary_suppression
        self.boundary_width = boundary_width
        self.boundary_iteration = boundary_iteration

        self.first_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
        self.first_conv.weight = torch.nn.Parameter(torch.ones_like((self.first_conv.weight)))
        self.first_conv.weight.requires_grad = False

        self.dilated_smoothing = dilated_smoothing
        sigma = 1.0
        size = 7
        gaussian_kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
        gaussian_kernel/= np.sum(gaussian_kernel)
        gaussian_kernel = torch.Tensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        self.second_conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, dilation=self.dilation_size, bias=False)
        self.second_conv.weight = torch.nn.Parameter(gaussian_kernel)
        self.second_conv.weight.requires_grad = False

        # NOTE(shjung13): These are for obtaining non-boundary masks
        # We calculate the boundary mask by subtracting the eroded prediction map from the dilated one
        # These are filters for erosion and dilation (L1)
        self.selem = torch.ones((3, 3))
        self.selem_dilation = torch.FloatTensor(ndi.generate_binary_structure(2, 1))

        # NOTE(shjung13): Dilation filters to expand the boundary maps (L1)
        d_k1 = torch.zeros((1, 1, 2 * 1 + 1, 2 * 1 + 1))
        d_k2 = torch.zeros((1, 1, 2 * 2 + 1, 2 * 2 + 1))
        d_k3 = torch.zeros((1, 1, 2 * 3 + 1, 2 * 3 + 1))
        d_k4 = torch.zeros((1, 1, 2 * 4 + 1, 2 * 4 + 1))
        d_k5 = torch.zeros((1, 1, 2 * 5 + 1, 2 * 5 + 1))
        d_k6 = torch.zeros((1, 1, 2 * 6 + 1, 2 * 6 + 1))
        d_k7 = torch.zeros((1, 1, 2 * 7 + 1, 2 * 7 + 1))
        d_k8 = torch.zeros((1, 1, 2 * 8 + 1, 2 * 8 + 1))
        d_k9 = torch.zeros((1, 1, 2 * 9 + 1, 2 * 9 + 1))

        self.d_ks = {1: d_k1, 2: d_k2, 3: d_k3, 4: d_k4, 5: d_k5, 6: d_k6, 7: d_k7, 8: d_k8, 9: d_k9}
        for k, v in self.d_ks.items():
            v[:,:,k,k] = 1
            for i in range(k):
                v = dilation(v, self.selem_dilation)
            self.d_ks[k] = v.squeeze(0).squeeze(0)

    def find_boundaries(self, label):
        """
        Calculate boundary mask by getting diff of dilated and eroded prediction maps
        """
        assert len(label.shape) == 4
        boundaries = (dilation(label.float(), self.selem_dilation.to(label.device)) != erosion(label.float(), self.selem.to(label.device))).float()
        return boundaries

    def expand_boundaries(self, boundaries, r=0):
        """
        Expand boundary maps with the rate of r
        """
        if r == 0:
            return boundaries
        expanded_boundaries = dilation(boundaries, self.d_ks[r].to(boundaries.device))
        return expanded_boundaries

    def forward(self, x, prediction):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(prediction.shape) == 3:
            prediction = prediction.unsqueeze(1)
        x_size = x.size()
        # B x 1 x H x W
        assert len(x.shape) == 4
        assert len(prediction.shape) == 4
        out = x
        if self.boundary_suppression:
            # obtain the boundary map of width 2 by default
            # this can be calculated by the difference of dilation and erosion
            boundaries = self.find_boundaries(prediction)
            expanded_boundaries = None
            if self.boundary_iteration != 0:
                assert self.boundary_width % self.boundary_iteration == 0
                diff = self.boundary_width // self.boundary_iteration
            for iteration in range(self.boundary_iteration):
                if len(out.shape) != 4:
                    out = out.unsqueeze(1)
                prev_out = out
                # if it is the last iteration or boundary width is zero
                if self.boundary_width == 0 or iteration == self.boundary_iteration - 1:
                    expansion_width = 0
                # reduce the expansion width for each iteration
                else:
                    expansion_width = self.boundary_width - diff * iteration - 1
                # expand the boundary obtained from the prediction (width of 2) by expansion rate
                expanded_boundaries = self.expand_boundaries(boundaries, r=expansion_width)
                # invert it so that we can obtain non-boundary mask
                non_boundary_mask = 1. * (expanded_boundaries == 0)

                f_size = 1
                num_pad = f_size

                # making boundary regions to 0
                x_masked = out * non_boundary_mask
                x_padded = nn.ReplicationPad2d(num_pad)(x_masked)

                non_boundary_mask_padded = nn.ReplicationPad2d(num_pad)(non_boundary_mask)

                # sum up the values in the receptive field
                y = self.first_conv(x_padded)
                # count non-boundary elements in the receptive field
                num_calced_elements = self.first_conv(non_boundary_mask_padded)
                num_calced_elements = num_calced_elements.long()

                # take an average by dividing y by count
                # if there is no non-boundary element in the receptive field,
                # keep the original value
                avg_y = torch.where((num_calced_elements == 0), prev_out, y / num_calced_elements)
                out = avg_y

                # update boundaries only
                out = torch.where((non_boundary_mask == 0), out, prev_out)
                del expanded_boundaries, non_boundary_mask

        # second stage; apply dilated smoothing
        if self.dilated_smoothing:
            out = nn.ReplicationPad2d(self.dilation_size * 3)(out)
            out = self.second_conv(out)

        return out
