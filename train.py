import sys

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import optim

import pytorch_ssim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def lr_lambda(epoch: int):
        if epoch < 2000:
            return 1
        else:
            if epoch < 4000:
                return 0.5
            else:
                return 0.1
            
def generate_kaleidoscope_pattern(image: torch.Tensor) -> torch.Tensor:
    """
    This function can generate 8 axis symmetric output based on input image
    """
    output_list = []
    axis_N = 6
    for i in range(axis_N):
        angle = 360 / float(axis_N) * i
        output_list.append(F.rotate(image, angle))
        output_list.append(F.rotate(F.hflip(image), angle))
    # output_list.append(image)
    # output_list.append(F.vflip(image))
    # output_list.append(image)
    return torch.mean(torch.cat(output_list, dim = 0), dim = 0, keepdim = True)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='path of target image you want to generate')
    parser.add_argument('--symmetric', action = 'store_true', help='path of target2 image you want to generate')
    parser.add_argument('--ssim_limit', type=float, default = 0.9, help='threshold of ssim score of generated images and target image')
    
    args = parser.parse_args()

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    target = torch.tensor(cv2.resize(np.array(Image.open(args.target))[:, :, 0:3] / (255.0), (600, 600))).unsqueeze(0).to(torch.float)
    pattern1 = torch.tensor(np.array(Image.open('pattern/pattern7.jpg'))[:, :, 0:3] / (255.0)).unsqueeze(0).to(torch.float)
    pattern2 = torch.tensor(np.array(Image.open('pattern/pattern3.jpg'))[:, :, 0:3] / (255.0)).unsqueeze(0).to(torch.float)
    pattern3 = torch.tensor(np.array(Image.open('pattern/pattern4.jpg'))[:, :, 0:3] / (255.0)).unsqueeze(0).to(torch.float)

    target = target.to(device)

    loss_weights = [3, 1, 1, 1]

    pattern1 = pattern1.to(device)
    pattern2 = pattern2.to(device)
    pattern3 = pattern3.to(device)

    _, h, w, c = target.shape

    c1 = torch.rand((1, h, w, c), requires_grad = True).to(torch.float)
    c2 = torch.rand((1, h, w, c), requires_grad = True).to(torch.float)
    c3 = torch.rand((1, h, w, c), requires_grad = True).to(torch.float)

    imgs = [c1, c2, c3]

    def forward(c1, c2, c3, gen = False):
        if gen is True:
            c1 = generate_kaleidoscope_pattern(c1)
            c2 = generate_kaleidoscope_pattern(c2)
            c3 = generate_kaleidoscope_pattern(c3)
        return torch.mean(
            torch.cat(
                [
                    torch.clamp(c1, 0, 1),
                    torch.clamp(c2, 0, 1),
                    torch.clamp(c3, 0, 1)
                ], 
                dim = 0
            ),
            dim = 0,
            keepdim = True
        )
        return torch.mul(
            torch.clamp(c1, 0, 1),
            torch.mul(
                torch.clamp(c2, 0, 1), torch.clamp(c3, 0, 1)
            )
        )
    
    
    with torch.no_grad():
        out = forward(c1.to(device), c2.to(device), c3.to(device))
    target_ssim = pytorch_ssim.ssim(out, target).item()
    print("Initial ssim:", target_ssim)
    ssim_loss = pytorch_ssim.SSIM()

    optimizer = optim.Adam([c1, c2, c3], lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    iter = 0
    patience = 500
    count = 0
    prev_target_ssim = 0
    while target_ssim < args.ssim_limit:
        iter += 1
        optimizer.zero_grad()
        out = forward(c1.to(device), c2.to(device), c3.to(device), args.symmetric)
        target_ssim = ssim_loss(target, out)

        if target_ssim <= prev_target_ssim or abs(target_ssim - prev_target_ssim) < 1e-5:
            count += 1
            if count > patience:
                break
        else:
            count = 0
        prev_target_ssim = target_ssim
        ssim_out = - loss_weights[0] * target_ssim\
                   - loss_weights[2] * ssim_loss(c2.to(device), pattern2)\
                   - loss_weights[1] * ssim_loss(c1.to(device), pattern1)\
                   - loss_weights[3] * ssim_loss(c3.to(device), pattern3)
        ssim_out /= sum(loss_weights)
        # ssim_out = - target_ssim
        ssim_value = - ssim_out.item()
        print(f"iter: {iter:05}, ssim_value: {target_ssim:.3f}, total_ssim: {ssim_value:.3f}, lr: {get_lr(optimizer)}", end = '\r')
        # f, axarr = plt.subplots(1, 4)
        # f.set_size_inches(10, 10)
        # for i in range(3):
        #     axarr[i].imshow(torch.clamp(imgs[i], 0, 1).squeeze().detach().numpy())
        #     axarr[i].get_xaxis().set_visible(False)
        #     axarr[i].get_yaxis().set_visible(False)
        # axarr[3].imshow(torch.clamp(out, 0, 1).squeeze().detach().numpy())
        # axarr[3].get_xaxis().set_visible(False)
        # axarr[3].get_yaxis().set_visible(False)
        # plt.show()
        ssim_out.backward()
        optimizer.step()
        scheduler.step()
    print()

    for i in range(3):
        if args.symmetric:
            plt.imsave(f"glass_paper_{i+1}.png", torch.clamp(generate_kaleidoscope_pattern(imgs[i]), 0, 1).squeeze().detach().numpy())
            plt.imsave(f"layer_{i+1}.png", torch.clamp(imgs[i], 0, 1).squeeze().detach().numpy())
        else:
            plt.imsave(f"layer_{i+1}.png", torch.clamp(imgs[i], 0, 1).squeeze().detach().numpy())
    plt.imsave('target.png', out.squeeze().detach().cpu().numpy())

if __name__ == '__main__':
    main()
