import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet import *
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        
    )

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

c = 16

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
        self.deformconv1 = DeformConv2d(c, 16, 3, 1, 1)
        self.deformconv2 = DeformConv2d(2*c, 2*c, 3, 1, 1)
        self.deformconv3 = DeformConv2d(4*c, 4*c, 3, 1, 1)
        self.deformconv4 = DeformConv2d(8*c, 8*c, 3, 1, 1)

    def flow_transform(self, flow):
        f = torch.empty((flow.shape[0], 18, flow.shape[2],flow.shape[3]), requires_grad=True)
        f = f.to(device)
        f = flow.repeat(1, 9, 1, 1)
        return f

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_t = self.flow_transform(flow)
        f1 = self.deformconv1(x, flow_t)
        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_t = self.flow_transform(flow)
        f2 = self.deformconv2(x, flow_t)
        
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_t = self.flow_transform(flow)
        f3 = self.deformconv3(x, flow_t)
        
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow_t = self.flow_transform(flow)
        f4 = self.deformconv4(x, flow_t)
        return [f1, f2, f3, f4]

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = Conv2(6, 2*c)
        self.down1 = Conv2(3*c, 4*c)
        self.down2 = Conv2(9*c, 9*c)
        self.down3 = Conv2(19*c, 19*c)
        self.up0 = deconv(39*c, 8*c)
        self.up1 = deconv(17*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 1, 1)

    def forward(self, warped_gt, lr, attn, c0):
        s0 = self.down0(torch.cat((warped_gt, lr), 1))
        s1 = self.down1(torch.cat((s0, c0[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], attn[0]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], attn[1]), 1))
        x = self.up0(torch.cat((s3, c0[3], attn[2]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x