import torch
import torch.nn.functional as F
from math import exp
from tools.args_fusion import args
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def mssim(img1, img2, window_size=11, val_range=None):
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

    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def std(img, window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) > 0, zero, one)
    ssim = map1 * ssim_ir + map2 * ssim_vi

    return ssim.mean()


def L_con_ssim(ir, vis, fuse):
    l_ssim = 1 - final_ssim(ir, vis, fuse)
    return l_ssim

def L_con_sal(ir, vis, sal_ir, sal_vis, fuse):
    l_sal = F.l1_loss(fuse, sal_ir * ir) + F.l1_loss(fuse, sal_vis * vis)
    return l_sal

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

def L_con_grad(ir, vis, fuse, device=None):

    sobelconv = Sobelxy(device)
    vi_grad_x, vi_grad_y = sobelconv(vis)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fuse)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
    return loss_grad

def L_con(ir, vis, fuse, sal_ir, sal_vis, device, alpha, beta, gamma):
    loss_sal = L_con_sal(ir, vis, sal_ir, sal_vis, fuse)
    loss_grad = L_con_grad(ir, vis, fuse, device)
    loss_ssim = L_con_ssim(ir, vis, fuse)
    l_con = alpha * loss_sal + beta * loss_grad + gamma * loss_ssim
    return l_con

def L_adv_G(score_g_ir, score_g_vis):
    l_adv_g_ir = (-torch.log(score_g_ir + args.eps)).mean()
    l_adv_g_vis = (-torch.log(score_g_vis + args.eps)).mean()
    l_adv_g = l_adv_g_ir + l_adv_g_vis
    return l_adv_g

def L_G(ir, vis, fuse, sal_ir, sal_vis, device, alpha, beta, gamma, score_g_ir, score_g_vis, lamda):
    l_g = L_adv_G(score_g_ir, score_g_vis) + lamda * L_con(ir, vis, fuse, sal_ir, sal_vis, device, alpha, beta, gamma)
    return l_g

def L_D_ir(score_ir, score_g_ir):
    l_d_ir = (-torch.log(score_ir + args.eps)).mean()
    l_d_g_ir = (-torch.log(1.0 - score_g_ir + args.eps)).mean()
    d_ir_loss = l_d_ir + l_d_g_ir
    return d_ir_loss

def L_D_vis(score_vis, score_g_vis):
    l_d_vis = (-torch.log(score_vis + args.eps)).mean()
    l_d_g_vis = (-torch.log(1.0 - score_g_vis + args.eps)).mean()
    d_vis_loss = l_d_vis + l_d_g_vis
    return d_vis_loss

