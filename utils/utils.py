import cv2
import numpy as np
import math
import torch

import torch.nn.functional as F
from math import exp
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result * 255

def convert_module(param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." 
        }
    
def convert_to_numpy(img):
    result = img.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    return result * 255
    
def psnr_calc(img0, img1):
    assert img0.shape == img1.shape, "Please provide tensor with same size"
    if((img0 - img1) * (img0 - img1)).mean() == 0:
        return 0.0
    else:
        return -10 * math.log10(((img0 - img1) * (img0 - img1)).mean())

def padding_swin(img, channel_size, window_size):
    assert len(img.shape) == 4, "Please provide a 4D tensor"
    
    b, c, h, w = img.shape
    pad_value = channel_size * window_size
    h_new = math.ceil(h / pad_value) * pad_value
    w_new = math.ceil(w / pad_value) * pad_value
    slice_h = int(((h_new-h)/2))
    slice_w = int(((w_new-w)/2))

    if slice_h != 0:
        img_r = torch.zeros(b, c, h_new, w_new)
        img_r[:, :, slice_h:-slice_h, slice_w:-slice_w] = img
        return img_r.to(device)
    else:
        return img

def crop_swin(img, h, w):
    B, _, H, W = img.shape
    if (H - h) == 1 and W - w == 1:
        img = img[:, :, 1:, 1:]
    else:
        h_diff = int(((H) - h) / 2)
        w_diff = int(((W) - w) / 2)
        if h_diff == 0 and w_diff == 0:
            return img
        else:
            if (H - h) % 2 ==  1 and (W - w) % 2 == 0:
                img = img[:, :, h_diff + 1:-h_diff, w_diff:-w_diff]
            elif (H - h) % 2 ==  0 and (W - w) % 2 == 1:
                img = img[:, :, h_diff:-h_diff, w_diff + 1:-w_diff]
            elif (H - h) % 2 ==  1 and (W - w) % 2 == 1:
                img = img[:, :, h_diff + 1:-h_diff, w_diff + 1:-w_diff]
            else:
                img = img[:, :, h_diff:-h_diff, w_diff:-w_diff]
            return img

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret