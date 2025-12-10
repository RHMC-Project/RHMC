import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image
import os
import argparse
import sys
import glob
import os.path as osp

def rgb2y_matlab(x):
    """Convert RGB image to illumination Y in Ycbcr space in matlab way.
    -------------
    # Args
        - Input: x, byte RGB image, value range [0, 255]
        - Ouput: byte gray image, value range [16, 235] 

    # Shape
        - Input: (H, W, C)
        - Output: (H, W) 
    """
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255**2 / mse) 


def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)
     
    ssim_score = structural_similarity(gt_img, noise_img, gaussian_weights=True, 
            sigma=1.5, use_sample_covariance=False)
    return ssim_score

def calculate_psnr_ssim(args):
    test_dir = args.test_dir
    gt_dir = args.gt_dir
    gt_img_list = sorted(glob.glob(osp.join(args.gt_dir, '*.png')))
    test_img_list = sorted(glob.glob(osp.join(args.test_dir, '*.png')))
    print("len(gt_img_list), len(test_img_list)")
    print(len(gt_img_list), len(test_img_list))
    assert len(gt_img_list) == len(test_img_list), 'Test image numbers are different from gt images.' 
    psnr_score = 0
    ssim_score = 0
    for gt_name, test_name in zip(gt_img_list, test_img_list):
        gt_img = Image.open(os.path.join(gt_dir, gt_name))
        test_img = Image.open(os.path.join(test_dir, test_name))
        gt_img = np.array(gt_img)
        test_img = np.array(test_img)
        psnr_score += PSNR(gt_img, test_img)
        ssim_score += SSIM(gt_img, test_img)
    return psnr_score / len(gt_img_list), ssim_score / len(gt_img_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()
    calculate_psnr_ssim(args)

