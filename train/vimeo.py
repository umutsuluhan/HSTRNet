import sys
sys.path.append('.')
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import os

from tqdm import tqdm
from model.HSTRNet import HSTRNet
from utils.dataset import VimeoDataset, DataLoader
from utils.utils import image_show, convert_module, ssim_matlab


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model):

    dataset_train = VimeoDataset("train", args.dataset, device)
    train_data = DataLoader(
        dataset_train, batch_size=args.train_bs, num_workers=args.workers, drop_last=True, shuffle=True
    )

    dataset_val = VimeoDataset("validation", args.dataset, device)
    val_data = DataLoader(dataset_val, batch_size=args.val_bs, num_workers=args.workers, shuffle=False)

    len_val = dataset_val.__len__()

    L1_lossFn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Training...")
    print("Train data set size:" + str(train_data.__len__()))
    print("Val data set size:" + str(val_data.__len__()))

    # Below code is a test to check if validation works as expected.
    # print("Validation is starting")
    # psnr, ssim = validate(model, val_data, len_val, 1)
    # print(psnr)
    # print(ssim)

    start = time.time()

    loss = 0
    psnr_list = []
    ssim_list = []
    for epoch in range(args.epoch):
        model.ifnet.train()
        model.contextnet.train()
        model.attention.train()
        model.unet.train()
        loss = 0

        print("Epoch: ", epoch)

        for trainIndex, data in enumerate(tqdm(train_data)):
            model.ifnet.train()
            model.contextnet.train()
            model.attention.train()
            model.unet.train()            

            data = data.to(device, non_blocking=True) / 255.0
            data = data.float()
            
            gt_batch = data[:, :18]
            ref_batch = data[:, 18:36]
            lr_batch = data[:, 36:54]
            
            for i in range(6):
                gt = gt_batch[:, i * 3: i * 3 + 3]
                ref = ref_batch[:, i * 3: i * 3 + 3]
                lr = lr_batch[:, i * 3: i * 3 + 3]
            
                imgs = torch.cat((ref, lr), 1)
                optimizer.zero_grad()
                pred = model(imgs)
                L1_loss = L1_lossFn(pred, gt)
                L1_loss.backward()
                optimizer.step()
                loss += float(L1_loss.item())
                end = time.time()

            if trainIndex == (train_data.__len__() - 1):
                torch.save(model.ifnet.state_dict(), f"{args.save_folder}/HSTR_ifnet_" + str(epoch) + ".pkl")
                torch.save(model.contextnet.state_dict(), f"{args.save_folder}/HSTR_contextnet_" + str(epoch) + ".pkl")
                torch.save(model.attention.state_dict(), f"{args.save_folder}/HSTR_attention_" + str(epoch) + ".pkl")
                torch.save(model.unet.state_dict(), f"{args.save_folder}/HSTR_unet_" + str(epoch) + ".pkl")
                torch.save(optimizer.state_dict(), f"{args.save_folder}/HSTR_optimizer_" + str(epoch) + ".pkl")
                
                print("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, ssim = validate(model, val_data, len_val, 1)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    endVal = time.time()

                print(
                    " Loss: %0.6f  TrainExecTime: %0.1f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f "
                    % (loss / trainIndex, end - start, psnr, endVal - end, ssim)
                )
                start = time.time()

    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=args.workers, shuffle=False)

    psnr, ssim = validate(model, val_data_last, len_val, 1)
    print("When batch size=1 --> PSNR:" + str(psnr) + " SSIM:" + str(ssim))

def validate(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.contextnet.eval()
    model.attention.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []

    for valIndex, data in enumerate(tqdm(val_data)):
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
    
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            imgs = torch.cat((ref, lr), 1)

            pred = model(imgs)
            for i in range(int(pred.shape[0])):
                
                if(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean()) <= 0:
                    psnr = 0
                else:
                    psnr = -10 * math.log10(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean())
                ssim_ = float(ssim_matlab(pred[i: i+1,:], gt[i: i+1,:]))
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
    return np.mean(psnr_list), np.mean(ssim_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--dataset", default='', type=str)
    parser.add_argument("--save_folder", default='model_dict/mami', type=str)
    parser.add_argument("--epoch", default=101, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--train_bs", default=16, type=int)
    parser.add_argument("--val_bs", default=4, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()


    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')

    model = HSTRNet(device)

    try:
        train(args, model)
    except Exception as e:
        print("Unexpected exception! %s", e)
