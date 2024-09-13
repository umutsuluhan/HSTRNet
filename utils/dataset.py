import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms.functional as FF
from PIL import Image
import time
import math
import torch.nn as nn

from utils.utils import image_show
from utils.aug import homography_d, gaussian_noise, contrast, horizontal_flip, rotate

device = "cuda"
img_index = 1 

def aug(gt, ref, lr, h, w, mode):
    if mode == "train_cufed":
        print("Cropping is skipped")
    else:    
        ih, iw, _ = gt.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        gt_batch = gt[x:x+h, y:y+w, :]
        ref_batch = ref[x:x+h, y:y+w, :]
        lr_batch = lr[x:x+h, y:y+w, :]
    
        cropped_gt_batch = gt_batch
        cropped_ref_batch = ref_batch

    if mode == "train":
        gt_r_batch = np.zeros(gt_batch.shape)
        ref_r_batch = np.zeros(ref_batch.shape)
        lr_r_batch = np.zeros(lr_batch.shape)
        for i in range(6):

            gt = gt_batch[:, :, i * 3 : i * 3 + 3]
            ref = ref_batch[:, :, i * 3 : i * 3 + 3]
            lr = lr_batch[:, :, i * 3 : i * 3 + 3]

            # Augmentations specific to HSTR_Data (real-time data)
            lr = homography_d(lr, 0.0000000001)

            gt = Image.fromarray(gt.astype(np.uint8))
            ref = Image.fromarray(ref.astype(np.uint8))
            lr = Image.fromarray(lr.astype(np.uint8))

            lr = gaussian_noise(lr)

            lr = contrast(lr)

            #Applying horizontal flip with %20 probability 
            #-------------------------------------------------------------------
            p = random.uniform(0.0, 1.0)
            if(p < 0.2):
                gt, ref, lr = horizontal_flip(gt, ref, lr)
            #-------------------------------------------------------------------
    
            #Applying rotation
            #-------------------------------------------------------------------
            gt, ref, lr = rotate(gt, ref, lr)
            #-------------------------------------------------------------------
    
            gt = np.array(gt)
            ref = np.array(ref)
            lr = np.array(lr)
            
            gt_r_batch[:,:,i*3:i*3+3] = gt
            ref_r_batch[:,:,i*3:i*3+3] = ref
            lr_r_batch[:,:,i*3:i*3+3] = lr
    return gt_r_batch, ref_r_batch, lr_r_batch


def aug_vis(gt, ref, lr, h, w, mode):
    if mode == "train_cufed":
        print("Cropping is skipped")
    else:    
        ih, iw, _ = gt.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        gt = gt[x:x+h, y:y+w, :]
        ref = ref[x:x+h, y:y+w, :]
        lr = lr[x:x+h, y:y+w, :]

    if mode == "train":
        gt = Image.fromarray(gt.astype(np.uint8))
        ref = Image.fromarray(ref.astype(np.uint8))
        lr = Image.fromarray(lr.astype(np.uint8))
        

        #Applying horizontal flip with %20 probability 
        #-------------------------------------------------------------------
        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            gt, ref, lr = horizontal_flip(gt, ref, lr)
        #-------------------------------------------------------------------
    
        #Applying rotation
        #-------------------------------------------------------------------
        gt, ref, lr = rotate(gt, ref, lr)
        #-------------------------------------------------------------------

        gt = np.array(gt)
        ref = np.array(ref)
        lr = np.array(lr)
    return gt, ref, lr

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR)

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "sequences/")
        data_path_LR = os.path.join(self.data_root, "x4_down_sequences_lr/")
        train_path = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_path = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry

        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)
    
    def getimg(self, index):
        gt_path = self.meta_data_HR[index] + "/" + 'im1.png'
        ref_path = self.meta_data_HR[index] + "/" + 'im2.png'
        lr_path = self.meta_data_LR[index] + "/" + 'im1.png'
        
        gt = cv2.imread(gt_path)
        ref = cv2.imread(ref_path)
        lr = cv2.imread(lr_path)

        if self.dataset_name == 'train':
            for i in range(2,7):
                p = random.uniform(0.0, 1.0)
                if(p < 0.5):
                    gt_path = self.meta_data_HR[index] + "/" + "im" + str(i) + '.png'
                    ref_path = self.meta_data_HR[index] + "/" + "im" + str(i + 1) + '.png'
                    lr_path = self.meta_data_LR[index] + "/" + "im" + str(i) + '.png'
                    gt = np.concatenate((gt, cv2.imread(gt_path)), 2)
                    ref = np.concatenate((ref, cv2.imread(ref_path)), 2)
                    lr = np.concatenate((lr, cv2.imread(lr_path)), 2)
                else:
                    gt_path = self.meta_data_HR[index] + "/" + "im" + str(i + 1) + '.png'
                    ref_path = self.meta_data_HR[index] + "/" + "im" + str(i) + '.png'
                    lr_path = self.meta_data_LR[index] + "/" + "im" + str(i + 1) + '.png'
                    gt = np.concatenate((gt, cv2.imread(gt_path)), 2)
                    ref = np.concatenate((ref, cv2.imread(ref_path)), 2)
                    lr = np.concatenate((lr, cv2.imread(lr_path)), 2)
        elif self.dataset_name == 'validation':
            for i in range(2,7):
                gt_path = self.meta_data_HR[index] + "/" + "im" + str(i) + '.png'
                ref_path = self.meta_data_HR[index] + "/" + "im" + str(i + 1) + '.png'
                lr_path = self.meta_data_LR[index] + "/" + "im" + str(i) + '.png'
                gt = np.concatenate((gt, cv2.imread(gt_path)), 2)
                ref = np.concatenate((ref, cv2.imread(ref_path)), 2)
                lr = np.concatenate((lr, cv2.imread(lr_path)), 2)
        return gt, ref, lr


    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
        if self.dataset_name == 'train':
            gt, ref, lr = aug(gt, ref, lr, 128, 128, "train")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)
            return torch.cat((gt, ref, lr), 0)
        elif self.dataset_name == "validation":
            #gt, ref, lr = aug(gt, ref, lr, 256, 448, "test")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device)
            return torch.cat((gt, ref, lr), 0)

class VimeoSeptupletDataset45(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR)

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "sequences/")
        data_path_LR = os.path.join(self.data_root, "x4_down_sequences_lr/")
        train_path = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_path = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry

        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)
    
    def getimg(self, index):
        gt_path = self.meta_data_HR[index] + "/" + 'im4.png'
        ref_path = self.meta_data_HR[index] + "/" + 'im5.png'
        lr_path = self.meta_data_LR[index] + "/" + 'im4.png'
        
        gt = cv2.imread(gt_path)
        ref = cv2.imread(ref_path)
        lr = cv2.imread(lr_path)
       
        return gt, ref, lr


    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
        if self.dataset_name == 'train':
            gt, ref, lr = aug(gt, ref, lr, 128, 128, "train")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)
            return torch.cat((gt, ref, lr), 0)
        elif self.dataset_name == "validation":
            #gt, ref, lr = aug(gt, ref, lr, 256, 448, "test")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device)
            return torch.cat((gt, ref, lr), 0)

class VimeoTripletDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR) - 1

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "sequences/")
        data_path_LR = os.path.join(self.data_root, "vimeo_triplet_lr/sequences/")
        train_path = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_path = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry
            
        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)

    
    def getimg(self, index):
        
        gt_path = self.meta_data_HR[index] + "/" + 'im2.png'
        ref_path = self.meta_data_HR[index] + "/" + 'im3.png'
        lr_path = self.meta_data_LR[index] + "/" + 'im2.png'
        
        gt =  cv2.imread(gt_path)
        ref = cv2.imread(ref_path)
        lr = cv2.imread(lr_path)
        return gt, ref, lr


    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)

        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
        ref = torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device)
        lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device)
        return torch.cat((gt, ref, lr), 0)

class VizdroneDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448


    def __len__(self):
        return len(self.meta_data_HR) - 3
    
    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR_train = os.path.join(self.data_root, "train/HR/")
        data_path_LR_train = os.path.join(self.data_root, "train/LR/")
        data_path_HR_val = os.path.join(self.data_root, "val/HR/")
        data_path_LR_val = os.path.join(self.data_root,"val/LR/")
        train_path = os.path.join(self.data_root, 'original_train_list.csv')
        test_path = os.path.join(self.data_root, 'original_val_list.csv')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()
        
        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR_train  + entry + ".jpg"
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR_train  + entry + ".jpg"
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR_val + entry + ".jpg"
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR_val + entry + ".jpg"
            self.testlist_LR[i] = new_entry

        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)
    
    def padding(self, img):
        padding1_mult = math.floor(img.shape[1] / 32) + 1
        padding2_mult = math.floor(img.shape[2] / 32) + 1
        pad1 = (32 * padding1_mult) - img.shape[1]
        pad2 = (32 * padding2_mult) - img.shape[2]
        
        img = torch.unsqueeze(img, 0)
        
        padding = nn.ZeroPad2d((int(pad2/2), int(pad2/2), int(pad1/2), int(pad1/2)))
        img = img.float()
        img = padding(img)
        img = torch.squeeze(img, 0)
        
        return img

    def getimg(self, index):
        
        gt = cv2.imread(self.meta_data_HR[index])
        ref = cv2.imread(self.meta_data_HR[index + 1])
        lr = cv2.imread(self.meta_data_LR[index])
        
        return gt, ref, lr

    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
        if self.dataset_name == 'train':
            gt, ref, lr = aug_vis(gt, ref, lr, 128, 128, "train")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)
            return torch.cat((gt, ref, lr), 0)
        elif self.dataset_name == "validation":
            gt = self.padding(torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device))
            ref = self.padding(torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device))
            lr = self.padding(torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device))
        return torch.cat((gt, ref, lr), 0)


class MAMIDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR) - 3

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "HR/")
        data_path_LR = os.path.join(self.data_root, "LR/")
        train_path = os.path.join(self.data_root, 'train_list.txt')
        test_path = os.path.join(self.data_root, 'test_list.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry
        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)

    
    def getimg(self, index):
        gt = cv2.imread(self.meta_data_HR[index])
        ref = cv2.imread(self.meta_data_HR[index + 1])
        lr = cv2.imread(self.meta_data_LR[index])

        return gt, ref, lr


    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
       
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
        ref = torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device)
        lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device)
        return torch.cat((gt, ref, lr), 0)
