This repository contains the official implementation of the paper:

"Semantic Augmentation Variational Autoencoder for Unsupervised Anomaly Detection in Retinal OCT Images."

📁 Project Overview

SeAugVAE is a novel unsupervised anomaly detection framework tailored for retinal Optical Coherence Tomography (OCT) images. It integrates a variational autoencoder architecture with semantic augmentation strategies to achieve accurate pixel-wise anomaly detection without the need for labeled anomaly data, preprocessing procedures, or auxiliary tasks.

📦 Files Structure

This codebase includes:

SeAugVAE/

├── train.py              #  Training scripts for unsupervised learning on retinal OCT datasets

├── inference.py          #  Inference scripts for generating semantic-spatial anomaly attention maps that highlight the pixel-wise anomalous regions

├── network.py            #  Network architecture of the proposed SeAugVAE model

└── README.md             #  This file

The following tables report slice-level performance (mean ± std) on two datasets.

📊 Table 1. Slice-level evaluation performance on Edema dataset
| Method | DICE | ACC | SEN | SPE |
|--------|------|-----|-----|-----|
| WeakAnD | 0.5493 ± 0.0020 | 0.9338 ± 0.0013 | 0.6037 ± 0.0259 | 0.9675 ± 0.0023 |
| MKD | 0.3304 ± 0.0067 | 0.7884 ± 0.0093 | 0.6126 ± 0.0112 | 0.8048 ± 0.0110 |
| RD | 0.5887 ± 0.0035 | 0.9183 ± 0.0016 | 0.6868 ± 0.0089 | 0.9398 ± 0.0024 |
| IGD | 0.5349 ± 0.0017 | 0.9024 ± 0.0005 | 0.6596 ± 0.0073 | 0.9250 ± 0.0012 |
| SPADE | 0.5635 ± 0.0000 | 0.8691 ± 0.0000 | 0.8219 ± 0.0000 | 0.8745 ± 0.0000 |
| PatchCore | 0.5548 ± 0.0133 | 0.9133 ± 0.0020 | 0.6352 ± 0.0206 | 0.9391 ± 0.0008 |
| SimpleNet | 0.5127 ± 0.0310 | 0.8737 ± 0.0218 | 0.7690 ± 0.0552 | 0.8834 ± 0.0290 |
| Grad-VAE | 0.5474 ± 0.0033 | 0.9008 ± 0.0004 | 0.7046 ± 0.0078 | 0.9191 ± 0.0007 |
| VAE+G&L | 0.5322 ± 0.0022 | 0.9112 ± 0.0009 | 0.5932 ± 0.0043 | 0.9408 ± 0.0012 |
| SeAugVAE+G&L | 0.5974 ± 0.0006 | 0.9282 ± 0.0001 | 0.6258 ± 0.0010 | 0.9563 ± 0.0001 |

📊 Table 2. Slice-level evaluation performance on CNV dataset
| Method | DICE | ACC | SEN | SPE |
|--------|------|-----|-----|-----|
| WeakAnD | 0.5493 ± 0.0020 | 0.9338 ± 0.0013 | 0.6037 ± 0.0259 | 0.9675 ± 0.0023 |
| MKD | 0.3304 ± 0.0067 | 0.7884 ± 0.0093 | 0.6126 ± 0.0112 | 0.8048 ± 0.0110 |
| RD | 0.5887 ± 0.0035 | 0.9183 ± 0.0016 | 0.6868 ± 0.0089 | 0.9398 ± 0.0024 |
| IGD | 0.5349 ± 0.0017 | 0.9024 ± 0.0005 | 0.6596 ± 0.0073 | 0.9250 ± 0.0012 |
| SPADE | 0.5635 ± 0.0000 | 0.8691 ± 0.0000 | 0.8219 ± 0.0000 | 0.8745 ± 0.0000 |
| PatchCore | 0.5548 ± 0.0133 | 0.9133 ± 0.0020 | 0.6352 ± 0.0206 | 0.9391 ± 0.0008 |
| SimpleNet | 0.5127 ± 0.0310 | 0.8737 ± 0.0218 | 0.7690 ± 0.0552 | 0.8834 ± 0.0290 |
| Grad-VAE | 0.5474 ± 0.0033 | 0.9008 ± 0.0004 | 0.7046 ± 0.0078 | 0.9191 ± 0.0007 |
| VAE+G&L | 0.5322 ± 0.0022 | 0.9112 ± 0.0009 | 0.5932 ± 0.0043 | 0.9408 ± 0.0012 |
| SeAugVAE+G&L | 0.5974 ± 0.0006 | 0.9282 ± 0.0001 | 0.6258 ± 0.0010 | 0.9563 ± 0.0001 |



📬 Contact

For additional materials, pretrained weights, or further implementation details, please contact:

📧 [xueyingz@shu.edu.cn]
