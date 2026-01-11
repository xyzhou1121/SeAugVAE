from __future__ import print_function
import datetime
import cv2
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
import numpy as np
from network import SeAugVAE
from datasetloader_test import testImageDataset
from evaluate import evaluate
from scipy.ndimage import gaussian_filter

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformation
data_transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.Resize(224),

    ]),
    "test": transforms.Compose([
        transforms.Resize(224),
    ])}

# Dataset root
image_path = r"E:\PyCodes\Response\DataSet/"

# Batch size
batch_size = 1

# Test dataset
test_dataset = testImageDataset(root=image_path + "test",
                                transform=data_transform["test"])
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=False,
                                          num_workers=0)

# Model
modelV = SeAugVAE().to(device)


def get_spatial_maps(real_fea, fake_fea):
    """
    Compute spatial anomaly attention maps based on the difference between real and reconstructed feature maps.

    Args:
        real_fea (torch.Tensor): Original feature maps extracted from input images.
        fake_fea (torch.Tensor): Reconstructed feature maps from the reconstruction images.

    Returns:
        spatial_map (torch.Tensor):  Spatial anomaly attention map (1 x 1 x 224 x 224).
    """
    real_fea_b = real_fea[0].clone().squeeze()
    fake_fea_b = fake_fea[0].clone().squeeze()
    # Compute squared difference in the feature domain
    spatial_map = torch.mean(torch.pow((F.relu(real_fea_b) - F.relu(fake_fea_b)), 2), dim=0)
    # Resize to match image resolution
    resize = transforms.Resize(224).to(device)
    spatial_map = resize(spatial_map.unsqueeze(0).unsqueeze(0))
    return spatial_map


def get_semantic_maps(feature_maps, weights):
    """
    Generate semantic anomaly attention maps using learned attention weights.

    Args:
        feature_maps (torch.Tensor): Last-layer feature maps from the encoder.
        weights (torch.Tensor): Semantic attention weights.

    Returns:
        semantic_map (torch.Tensor): Attention-weighted semantic map (1 x 1 x 224 x 224).
    """
    weight = weights[0].clone()
    feature_map = feature_maps[0].clone()
    weight = weight[1:]
    weight = weight.unsqueeze(-1).unsqueeze(-1)
    # Weighted feature combination
    feature_map = weight * feature_map
    feature_map[feature_map < 0] = 0
    # Average across channels and resize
    feature_map = torch.mean(feature_map, dim=0).unsqueeze(0).unsqueeze(0)
    resize = transforms.Resize(224).to(device)
    feature_map = resize(feature_map)
    semantic_map = feature_map
    return semantic_map


def inference():
    """
    Perform inference on the test dataset using the trained SeAugVAE model.
    Computes anomaly score maps and evaluates the model using standard metrics.
    """
    model_weight = "SeAugVAE_NetV1000"
    model_weight_path = os.path.join(r"SeAugVAE\pths", model_weight)
    model_weight_path = model_weight_path + ".pth"
    modelV.load_state_dict(torch.load(model_weight_path, map_location='cuda:0'))

    for i, data in enumerate(test_loader):
        modelV.eval()
        # Load test data and move to device
        data, label_img = data['image'], data['label']
        data = data.to(device)

        # Input original image
        out, mu2, logvar2, feature_maps, weight, y = modelV(data, False)

        # Input reconstructed image
        out_fake, mu2, logvar2, _, y_fake = modelV(out)

        # Compute semantic and spatial anomaly attention maps
        _, masks = get_semantic_maps(feature_maps, weight)
        spatial_map = get_spatial_maps(y, y_fake)

        # Combine anomaly score maps and apply Gaussian smoothing
        anomaly_score = spatial_map * 0.002 + masks
        anomaly_score = gaussian_filter(np.array(anomaly_score.cpu()), sigma=4)

        rate = (i + 1) / len(test_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest: {:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")

    # Evaluate performance using standard metrics
    auc, prc, dice, acc, f1, sen, spe = evaluate(label_img, anomaly_score, "local")
    print(
        '====> epoch: {}, AUROC: {:.4f}, AUPRC: {:.4f}, Dice: {:.4f}, ACC: {:.4f}, F1: {:.4f}, Sen: {:.4f}, Spe: {:.4f}'.format(
            0, auc, prc, dice, acc, f1, sen, spe))
    return auc, prc, dice


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    with torch.no_grad():
        inference()
        end_time = datetime.datetime.now()
        time = str(end_time - start_time)
        print("time：" + time[0:7])
        print("-" * 10)
        print(' ')




