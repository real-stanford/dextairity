import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq


def rotate_tensor2d(inputs, rotate_theta, offset=None, padding_mode='zeros', pre_padding=None):
    """rotate 2D tensor counter-clockwise
    Args:
        inputs: torch tensor, [N, C, W, H]
        rotate_theta: ndarray,[N]
        offset: None or ndarray, [2, N]
        padding_mode: "zeros" or "border"
        pre_padding: None of float. the valud used for pre-padding such that width == height
    Retudn:
        outputs: rotated tensor
    """
    device = inputs.device

    if pre_padding is not None:
        lr_pad_w = int((np.max(inputs.shape[2:])-inputs.shape[3])/2)
        ud_pad_h = int((np.max(inputs.shape[2:])-inputs.shape[2])/2)
        add_pad = nn.ConstantPad2d((lr_pad_w,lr_pad_w,ud_pad_h,ud_pad_h),0.0).to(device)
        inputs = add_pad(inputs)

    const_zeros = np.zeros(len(rotate_theta))
    affine = np.asarray([[np.cos(rotate_theta), -np.sin(rotate_theta), const_zeros],
                         [np.sin(rotate_theta), np.cos(rotate_theta), const_zeros]])
    affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
    flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
    outputs = F.grid_sample(inputs, flow_grid, padding_mode=padding_mode, align_corners=True)

    if offset is not None:
        const_ones = np.ones(len(rotate_theta))
        affine = np.asarray([[const_ones, const_zeros, offset[0]],
                            [const_zeros, const_ones, offset[1]]])
        affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
        flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
        outputs = F.grid_sample(outputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if pre_padding is not None:
        outputs = outputs[:,:,ud_pad_h:(outputs.shape[2]-ud_pad_h),
                              lr_pad_w:(outputs.shape[3]-lr_pad_w)]
    return outputs


def MLP(channels, last_relu=True):
    module_list = list()
    for i in range(1, len(channels)):
        if i == len(channels) - 1 and not last_relu:
            module = Lin(channels[i - 1], channels[i])
        else:
            module = Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        
        module_list.append(module)
    return Seq(*module_list)


class ConvBlock1D(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()

        self.conv = torch.nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm)
        self.norm = torch.nn.BatchNorm1d(planes) if norm else None
        self.relu = torch.nn.LeakyReLU(inplace=True) if relu else None
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else torch.nn.functional.interpolate(out, scale_factor=2, mode='linear', align_corners=True)

        return out


class ConvBlock2D(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()

        self.conv = torch.nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm)
        self.norm = torch.nn.BatchNorm2d(planes) if norm else None
        self.relu = torch.nn.LeakyReLU(inplace=True) if relu else None
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out


class ResBlock2D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, last_activation=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.activation2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation1(out)

        return out