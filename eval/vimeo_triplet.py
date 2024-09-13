import argparse
import sys
sys.path.append('.')
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import logging

from tqdm import tqdm
from model.HSTRNet import HSTRNet
from utils.dataset import VimeoTripletDataset, DataLoader
from utils.utils import ssim_matlab


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_data):
    psnr_list = []
    ssim_list = []
    total_times = []

    for valIndex, data in enumerate(tqdm(val_data)):
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
            
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            imgs = torch.cat((ref, lr), 1)
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = model(imgs, gt)
            end.record()
            torch.cuda.synchronize()
            total_times.append(start.elapsed_time(end))
    	
            psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            ssim_ = float(ssim_matlab(pred, gt))
            psnr_list.append(psnr)
            ssim_list.append(ssim_)

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(total_times)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = HSTRNet(device)

    dataset_val = VimeoTripletDataset("validation", "/home/ortak/mughees/datasets/vimeo_triplet/", device)
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    model.ifnet.load_state_dict(torch.load('./pretrained/vimeo/HSTR_ifnet_62.pkl', map_location=device))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./pretrained/vimeo/HSTR_contextnet_62.pkl', map_location=device))
    model.contextnet.eval()

    model.attention.load_state_dict(torch.load('./pretrained/vimeo/HSTR_attention_62.pkl', map_location=device))
    model.attention.eval()
    
    model.unet.load_state_dict(torch.load("./pretrained/vimeo/HSTR_unet_62.pkl", map_location=device))
    model.unet.eval()

    psnr,  ssim_, exec_time = validate(model, val_data_last) 

    print("PSNR:", psnr)
    print("SSIM:", ssim_)
    print("Average Time:", exec_time)
