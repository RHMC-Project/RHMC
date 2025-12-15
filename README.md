# RHMC: Region-Aware Hybrid Mamba-CNN with Symlet Transform for Face Super-Resolution

## 🖼️ Architecture Visualization 

<p align="center">
  <img src="https://github.com/RHMC-Project/RHMC/blob/main/figure1.png">
</p>

<p align="center">
  <img src="https://github.com/RHMC-Project/RHMC/blob/main/figure2.png">
</p>


## 🚀 Abstract

Face Super-Resolution (FSR) aims to reconstruct high-resolution facial images with photo-realistic fidelity and identity consistency from low-resolution inputs. Recent advances show that the **Mamba** architecture is effective in modeling global facial structures, yet its direct application to FSR faces two major limitations.

1.  **Non-Adaptive Processing:** Standard Mamba assigns independent SSM kernels to each channel and uniformly processes all facial regions, failing to adaptively handle region-specific structural and high-frequency details.
2.  **Weak Local Perception:** Its intrinsic lack of local perception hampers the modeling of fine-grained 2D textures, leading to insufficient representation of local pixel relationships.

To overcome these challenges, we propose **RHMC**, a **R**egion-Aware **H**ybrid **M**amba--**C**NN with Symlet Transform tailored for FSR.

* Mamba is leveraged to capture **global structural dependencies**, while a parallel CNN branch compensates for its weak **local texture modeling**.
* To enable adaptive fusion across facial regions, we design a **Region-aware Mixture-of-Experts (RMoE)** that dynamically blends global structural cues and local textural features in a region-aware manner.
* We further introduce a **Feature Aggregation Module (FAM)** that employs efficient channel attention to recalibrate feature importance globally, preserving critical facial information and mitigating local detail degradation caused by Mamba's sequential modeling.
* Moreover, inspired by facial bilateral symmetry, a **Symlet Fusion Block** is developed to modulate and integrate contour and texture features in the frequency domain, enhancing realism and suppressing structural artifacts.

Extensive experiments demonstrate that RHMC substantially outperforms existing state-of-the-art FSR methods in both reconstruction quality and perceptual fidelity. Code will be publicly available at [https://github.com/RHMC-Project/RHMC](https://github.com/RHMC-Project/RHMC).

---

## 🌟 Project Overview

This repository provides the official PyTorch implementation of the **RHMC** model/method proposed in our paper:

**[RHMC: Region-Aware Hybrid Mamba-CNN with Symlet Transform for Face Super-Resolution]** (Placeholder Title)

The code includes scripts for:
* Model training and testing.
* Evaluation using common metrics (**PSNR, SSIM, LPIPS, VIF**).

## 💡 Key Contributions

* We propose the novel **RHMC** module, a **Hybrid Mamba-CNN** architecture, which adaptively fuses global structural information (via Mamba) and local textures (via CNN) using a **Region-aware Mixture-of-Experts (RMoE)**.
* We introduce a **Symlet Fusion Block** that leverages facial bilateral symmetry and frequency-domain analysis to enhance realism and suppress artifacts.
* RHMC achieves competitive state-of-the-art results on popular face/natural image datasets for Super-Resolution.

---




# Recommended execution using shell script
bash train.sh

# Or, run the python script directly, passing configurations/parameters as needed:
python train.py --config model/RHMC/config.yaml


# Recommended execution using shell script
bash train.sh

# Or, run the python script directly, passing configurations/parameters as needed:
python train.py --config model/RHMC/config.yaml


# Recommended execution using shell script
bash test.sh

# Or, run the python script directly:
python test.py --checkpoint path/to/checkpoint.pth --config model/RHMC/config.yaml

# Calculate PSNR and SSIM
python metrics/calculate_psnr_ssim.py --pred_dir experiments/preds --gt_dir experiments/gts

# Calculate LPIPS (Perceptual Metric)
python metrics/calculate_lpips.py --pred_dir experiments/preds --gt_dir experiments/gts
