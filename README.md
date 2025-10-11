This repository contains the official implementation of the paper:

"Semantic Augmentation Variational Autoencoder for Unsupervised Anomaly Detection in Retinal OCT Images."

📁 Project Overview

SeAugVAE is a novel unsupervised anomaly detection framework tailored for retinal Optical Coherence Tomography (OCT) images. It integrates a variational autoencoder architecture with semantic augmentation strategies to achieve accurate pixel-wise anomaly detection without the need for labeled anomaly data, preprocessing procedures, or auxiliary tasks.

📦 Files Structure

This codebase includes:

SeAugVAE/
├── train.py          # TTraining scripts for unsupervised learning on retinal OCT datasets

├── inference.py      # Inference scripts for generating semantic-spatial anomaly attention maps that highlight the pixel-wise anomalous regions

├── network.py        # Network architecture of the proposed SeAugVAE model

└── README.md         # This file


📬 Contact

For additional materials, pretrained weights, or further implementation details, please contact:

📧 [xueyingz@shu.edu.cn]
