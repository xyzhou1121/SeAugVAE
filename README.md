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

🧠 Pretrained Model Weights

The pretrained model weights for the Cirrus and Spectralis datasets are publicly available through Baidu Netdisk.

Download: [Baidu Netdisk](https://pan.baidu.com/s/1IChYbmC1tDug2A25cSxN8g?pwd=5qkr 提取码: 5qkr)

The following tables report slice-level performance (mean ± std) on two datasets.

📊 Table 1. Slice-level evaluation performance on Edema dataset
| Method | DICE | ACC | SEN | SPE |
|--------|------|-----|-----|-----|
| WeakAnD | 0.7082 ± 0.0011 | 0.9262 ± 0.0006 | 0.7153 ± 0.0078 | 0.9578 ± 0.0013 |
| MKD | 0.4889 ± 0.0031 | 0.7837 ± 0.0039 | 0.7123 ± 0.0054 | 0.7958 ± 0.0054 |
| RD | 0.7286 ± 0.0105 | 0.9255 ± 0.0157 | 0.7614 ± 0.0388 | 0.9487 ± 0.0154 |
| IGD | 0.6559 ± 0.0030 | 0.8837 ± 0.0018 | 0.7634 ± 0.0019 | 0.9041 ± 0.0023 |
| SPADE | 0.7097 ± 0.0000 | 0.8768 ± 0.0000 | 0.8704 ± 0.0000 | 0.8781 ± 0.0000 |
| PatchCore | 0.6859 ± 0.0031 | 0.9008 ± 0.0010 | 0.7454 ± 0.0044 | 0.9272 ± 0.0010 |
| SimpleNet | 0.6178 ± 0.0242 | 0.8534 ± 0.0182 | 0.8105 ± 0.0221 | 0.8607 ± 0.0250 |
| Grad-VAE | 0.7092 ± 0.0006 | 0.9031 ± 0.0002 | 0.8135 ± 0.0012 | 0.9184 ± 0.0003 |
| VAE+G&L | 0.6719 ± 0.0014 | 0.8960 ± 0.0005 | 0.7332 ± 0.0012 | 0.9237 ± 0.0005 |
| SeAugVAE+G&L | 0.7319 ± 0.0007 | 0.9193 ± 0.0002 | 0.7590 ± 0.0011 | 0.9465 ± 0.0002 |


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
