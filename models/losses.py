import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF

shape = (256, 256)
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map*mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



def corr_loss(image_vis, img_ir, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_vis, img_ir, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss


class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)
def Fusion_loss(vi, ir, fu, weights=[1, 10, 1], device=None):
    sobelconv = Sobelxy(device)
    x_in_max=torch.max(vi,ir)
    y_grad=sobelconv(vi)
    ir_grad=sobelconv(ir)
    loss_in=F.l1_loss(x_in_max,fu)
    generate_img_grad=sobelconv(fu)
    x_grad_joint=torch.max(y_grad,ir_grad)
    loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
    # loss_ssim = (torch.ones(1) - ssim(fu, vi) + torch.ones(1) - ssim(fu, ir)).to(device)
    loss_corr = corr_loss(vi, ir, fu) 
    # loss_total = weights[0] * loss_corr + weights[1] * loss_grad + weights[2] * loss_in
    return loss_corr,loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
