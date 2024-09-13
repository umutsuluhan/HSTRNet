import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import argparse
import io
import torch.nn.functional as F


from torch.nn.parallel import DistributedDataParallel as DDP
from model.networks.warplayer import warp
from model.networks.IFNet import IFNet
from model.networks.UNet import ContextNet, FusionNet
from model.networks.attention import Swin_V2
from utils.utils import image_show


class HSTRNet(nn.Module):
    def __init__(self, device):
        super(HSTRNet, self).__init__()
        self.device = device
        
        self.ifnet = IFNet(self.device)
        self.contextnet = ContextNet()
        self.attention = Swin_V2(3)
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.contextnet.to(self.device)
        self.attention.to(self.device)
        self.unet.to(self.device)
    
    def forward(self, imgs, gt=None):
        ref = imgs[:, :3]
        lr = imgs[:, 3:6]
        # image_show(ref)
        # image_show(lr)
        
        # ref-->t+1, lr-->t
        start_time_rife = time.time()
        _, flow = self.ifnet(torch.cat((lr, ref), 1))
        rife_time = time.time() - start_time_rife
        
        # 0.5 --> lr, 0.5 -->ref
        # what we need --> gt --> ref (0-->1)
        # (0.5 --> ref) * 2 = 0-->ref (gt-->ref)
        
        f_0_1 = flow[:, 2:4] * 2
        f_0_1 = F.interpolate(f_0_1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_gt = warp(ref, f_0_1, self.device)
        # Warped gt and warped gt2 gives same result, using warped gt2 directly might give better performance
        # psnr = -10 * math.log10(((gt - warped_gt) * (gt - warped_gt)).mean())
        #print(psnr)
        
        start_time_context = time.time()
        c0 = self.contextnet(ref, f_0_1)
        context_time = time.time() - start_time_context

        start_time_attention = time.time()
        attn = self.attention(lr, ref)
        attention_time = time.time() - start_time_attention

        start_time_fusion = time.time()        
        refine_output = self.unet(warped_gt, lr, attn, c0)
        fusion_time = time.time() - start_time_fusion
        
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_gt * mask + lr * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        return pred
        # return pred, rife_time, context_time, fusion_time, 0
