# RHMC: Region-Aware Hybrid Mamba-CNN with Symlet Transform for Face Super-Resolution

This repository accompanies our paper published in **Computer Vision and Image Understanding**:

**RHMC: Region-Aware Hybrid Mamba-CNN with Symlet Transform for Face Super-Resolution**

Wen Xue, Haifeng Li, Shuyang Chu, Jian Xu, and Jingang Shi

📄 [Paper](https://doi.org/10.1016/j.cviu.2026.105013)

## 🖼️ Architecture

<p align="center">
  <img src="./figure1.png" alt="Overall architecture of RHMC" width="100%">
</p>

RHMC adopts an encoder–decoder architecture with three main stages:
**feature encoding**, **feature aggregation**, and **feature decoding**.

<p align="center">
  <img src="./figure2.png" alt="Key modules of RHMC" width="100%">
</p>

The network combines Mamba-based global modeling, CNN-based local feature
extraction, region-aware feature fusion, and Symlet-based frequency fusion
for face super-resolution.

## 🚀 Abstract

Face Super-Resolution (FSR) aims to reconstruct high-resolution facial
images with photo-realistic fidelity and identity consistency from
low-resolution inputs. Recent advances show that the Mamba architecture
is effective in modeling global facial structures, yet its direct
application to FSR faces challenges in region-adaptive reconstruction
and fine-grained local texture modeling.

To address these challenges, we propose **RHMC**, a **Region-Aware Hybrid
Mamba-CNN with Symlet Transform** tailored for FSR. Mamba captures global
structural dependencies, while a parallel CNN branch complements its
local texture modeling. A **Region-aware Mixture-of-Experts (RMoE)**
mechanism dynamically fuses structural and textural features according
to facial content.

We further introduce a **Feature Aggregation Module (FAM)** that uses
efficient channel attention to recalibrate feature responses, preserve
critical facial information, and mitigate local detail degradation.
Inspired by facial bilateral symmetry, a **Symlet Fusion Block (SFB)**
integrates shallow contour features and deep texture features in the
frequency domain, enhancing reconstruction realism and suppressing
structural artifacts.

Experiments on CelebA and Helen demonstrate improvements in reconstruction
quality and perceptual fidelity over the compared methods. Qualitative
evaluations on Solvay Conference images and SCface examine generalization
to real-world images, while face verification experiments on LFW assess
the preservation of identity-discriminative information.

## 💡 Key Contributions

- **Hybrid global and local modeling.**
  We propose RHMC, a hybrid Mamba-CNN framework that jointly models
  global facial structures and fine-grained local textures.

- **Region-aware adaptive fusion.**
  We design RMoE to adaptively fuse structural and textural features
  using spatial and channel-wise weights derived from facial content.

- **Feature aggregation for detail preservation.**
  We introduce FAM to recalibrate channel responses and mitigate
  information forgetting when modeling local facial details.

- **Symlet-based frequency fusion.**
  We develop SFB to integrate shallow contour features and deep texture
  features through frequency-domain fusion, improving visual realism
  and suppressing structural artifacts.

## 🧩 Main Components

| Component | Description                                                  |
| --------- | ------------------------------------------------------------ |
| **HMCB**  | Hybrid Mamba-CNN Block for global structure and local texture modeling. |
| **RMoE**  | Region-aware Mixture-of-Experts mechanism for adaptive feature fusion. |
| **RSSM**  | Residual State Space Module for long-range dependency modeling and feature refinement. |
| **FAM**   | Feature Aggregation Module for channel recalibration and facial detail preservation. |
| **STB**   | Symlet Transformation Block for extracting frequency-domain components. |
| **SFB**   | Symlet Fusion Block for integrating encoder and decoder features in the frequency domain. |

## 📊 Experimental Results

### ×8 Face Super-Resolution

RHMC is evaluated on **CelebA** and **Helen**, reconstructing
128 × 128 facial images from 16 × 16 low-resolution inputs.

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |  VIF ↑ |
| ------- | -----: | -----: | ------: | -----: |
| Helen   |  28.34 | 0.8332 |  0.1570 | 0.4615 |
| CelebA  |  28.48 | 0.8257 |  0.1532 | 0.4640 |

PSNR, SSIM, and VIF are evaluated on the Y channel of the YCbCr color space.

Compared with the second-best methods in our evaluation, RHMC improves
PSNR by **0.27 dB on Helen** and **0.20 dB on CelebA**.

### Computational Efficiency

| Parameters |  FLOPs | Inference Time |
| ---------: | -----: | -------------: |
|     10.6 M | 18.2 G |        42.5 ms |

The reported inference time is measured on a single NVIDIA A100 GPU
under the evaluation setting described in the paper.

### Real-World Evaluation

We conduct qualitative comparisons on:

- **Solvay Conference images:** historical facial imagery.
- **SCface:** surveillance images captured under real-world conditions.

### Face Verification

We evaluate reconstructed faces on **LFW** using a pretrained
**ArcFace** model to extract identity features.

| Method       | TAR at FAR = 0.01% |
| ------------ | -----------------: |
| RHMC         |             91.20% |
| HR reference |             99.34% |

RHMC achieves the highest verification performance among the FSR
methods compared in the paper.

## 🔬 Ablation Studies

We investigate the contributions of:

- The region-aware mixture-of-experts mechanism.
- The feature aggregation module.
- Frequency-domain fusion in STB and SFB.
- The dual-branch split strategy in SFB.
- The Symlet transform compared with Haar wavelets, Daubechies wavelets,
  DCT, and FFT.

Please refer to the paper for detailed experimental settings and results.

The complete implementation of RHMC, including training, evaluation, and inference scripts, will be publicly released in this repository.



## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{xue2026rhmc,
  title   = {{RHMC}: Region-Aware Hybrid {Mamba-CNN} with {Symlet}
             Transform for face super-resolution},
  author  = {Xue, Wen and Li, Haifeng and Chu, Shuyang
             and Xu, Jian and Shi, Jingang},
  journal = {Computer Vision and Image Understanding},
  year    = {2026},
  pages   = {105013},
  doi     = {10.1016/j.cviu.2026.105013}
}