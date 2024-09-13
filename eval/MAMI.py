import sys
sys.path.append('.')
import torch
import numpy as np
import random

from tqdm import tqdm
from model.HSTRNet import HSTRNet
from utils.dataset import MAMIDataset
from utils.dataset import DataLoader
from utils.utils import psnr_calc, ssim_matlab

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
    
            psnr = psnr_calc(pred, gt)
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
   
    dataset_val = MAMIDataset("validation", "/home/ortak/mughees/datasets/MAMI/test/", device)
    val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    model.ifnet.load_state_dict(torch.load('./pretrained/mami/HSTR_ifnet_68.pkl', map_location=device))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./pretrained/mami/HSTR_contextnet_68.pkl', map_location=device))
    model.contextnet.eval()

    model.attention.load_state_dict(torch.load('./pretrained/mami/HSTR_attention_68.pkl', map_location=device))
    model.attention.eval()
    
    model.unet.load_state_dict(torch.load("./pretrained/mami/HSTR_unet_68.pkl", map_location=device))
    model.unet.eval()
    psnr,  ssim_, exec_time = validate(model, val_data) 

    print("PSNR:", psnr)
    print("SSIM:", ssim_)
    print("Average Time:", exec_time)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
