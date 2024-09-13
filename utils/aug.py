import numpy as np
import cv2
import torch
import random
import torchvision.transforms.functional as FF

def homography(self, img):
    img = self.convert_to_numpy(img)
    homography = np.zeros((3,3))
    homography[0][0] = 1
    homography[1][1] = 1
    homography[2][2] = 1
    homography = homography + (0.0000000005 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography[2][2] = 1
    homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    homography_img = torch.from_numpy(np.transpose(homography_img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
    return homography_img
    
def homography_d(img, p):        
    homography = np.zeros((3,3))
    homography[0][0] = 1
    homography[1][1] = 1
    homography[2][2] = 1
    homography = homography + (p ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography[2][2] = 1
    homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    return homography_img

def gaussian_noise(img0):  
    """factor = random.uniform(0.0, 2.5)
    img0 = img0 + np.random.randn(*img0.shape) * factor + 0"""
    return FF.gaussian_blur(img0, (3,3), 0.7)

def contrast(img0):  
    factor = random.uniform(0.9, 1.1)
    img0 = FF.adjust_contrast(img0, factor)
    return img0

def horizontal_flip(img0, img1, img2):
    img0 = FF.hflip(img0)
    img1 = FF.hflip(img1)
    img2 = FF.hflip(img2)
    return img0, img1, img2

def rotate(img0, img1, img2):
    degree = random.uniform(-10.0, 10.0)
    rotated_img0 = img0.rotate(degree)
    rotated_img1 = img1.rotate(degree)
    rotated_img2 = img2.rotate(degree)
    return rotated_img0, rotated_img1, rotated_img2