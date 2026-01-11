
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


def weights_init(mod):
    """
    Custom weights initialization called on netV, netI and netF
    """
    classname = mod.__class__.__name__
    if classname.find('Conv2d') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Standard residual block (used in ResNet-18/34).

    Each block contains two 3x3 convolutional layers with batch normalization and ReLU activation.
    A shortcut (identity or downsampled) connection adds the residual mapping.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block (used in ResNet-50/101/152).

    Structure: 1x1 -> 3x3 -> 1x1 convolutions.
    Provides a deeper residual mapping with reduced computational cost.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
     Simplified ResNet backbone used as encoder in the SeAugVAE model.

     Adapted for single-channel (grayscale) OCT images.
     """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y1 = x
        x = self.layer2(x)
        y2 = x
        x = self.layer3(x)
        y3 = x
        x = self.layer4(x)
        y4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, y1, y2, y3, y4

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Return a ResNet-18 model adapted for grayscale input."""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class BasicTran(nn.Module):
    """
    Basic transposed convolution block for upsampling in the decoder.
    Consists of ConvTranspose2d + BatchNorm + ReLU.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicTran, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeAugVAE(nn.Module):
    """
    Semantic Augmentation Variational Autoencoder (SeAugVAE).

    Combines a ResNet-based encoder with a multi-stage transposed convolution decoder.
    Used for unsupervised anomaly detection in retinal OCT images.
    """
    def __init__(self):
        super(SeAugVAE, self).__init__()

        self.resnetmodel = resnet18()
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 256)

        self.decoder = nn.Sequential(
            BasicTran(256, 32, kernel_size=7, stride=1, padding=0),
            BasicTran(32, 64, kernel_size=3, stride=1, padding=1),
            BasicTran(64, 128, kernel_size=3, stride=1, padding=1),
            BasicTran(128, 64, kernel_size=4, stride=2, padding=1),
            BasicTran(64, 64, kernel_size=3, stride=1, padding=1),
            BasicTran(64, 32, kernel_size=4, stride=2, padding=1),
            BasicTran(32, 32, kernel_size=3, stride=1, padding=1),
            BasicTran(32, 32, kernel_size=4, stride=2, padding=1),
            BasicTran(32, 32, kernel_size=4, stride=2, padding=1),
            BasicTran(32, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def encode(self, x):
        h1, y1, y2, y3, y4 = self.resnetmodel(x)
        h1 = F.relu(h1)
        y2 = self.ap1(y2)
        y3 = self.ap1(y3)
        y = torch.cat((torch.cat((y1, y2), dim=1), y3), dim=1)
        return self.fc21(h1), self.fc22(h1), y4, y

    def calculatescore(self, yy, ormu):
        """Compute weight for each channel."""
        yy = F.adaptive_avg_pool2d(yy, (1, 1))
        yy = torch.flatten(yy, 1)
        yy = self.fc21(yy)
        score = torch.mean(torch.pow((F.relu(ormu) - F.relu(yy)), 2))
        return score

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample z = μ + σ * ε."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decode latent variable z into reconstructed image."""
        zz = self.fc3(z)
        h3 = F.relu(zz, inplace=True)
        v = h3.view(-1, 256, 1, 1)
        outc = self.decoder(v)
        return outc

    def in_score(self, mu, y):
        """
        Compute importance weights for feature maps.
        Used to estimate semantic contribution of each feature channel.
        """
        score = torch.zeros((y.size(0), 513))
        for b in range(y.size(0)):
            scores = []
            scores.append(torch.sum(F.relu(mu[b])).cpu().detach().numpy())
            for i in range(512):
                temp = y[b].clone().unsqueeze(dim=0)
                temp[:, i, :, :] = 0
                scorei = self.calculatescore(temp, mu[b])
                scores.append(scorei.cpu().detach().numpy())
            scores = np.array(scores)
            scores = torch.from_numpy(scores).type(torch.FloatTensor)
            scores = torch.unsqueeze(scores, dim=0)
            score[b] = scores
        return score


    def forward(self, x, iftrain=True):
        """
        Forward pass of the dVAE model.

        Args:
            x (torch.Tensor): Input OCT image.
            iftrain (bool): Whether in training mode (True) or inference mode (False).

        Returns:
            During training: (reconstructed image, μ, logσ², z)
            During inference: (reconstructed image, μ, logσ², last-layer features, importance weights)
        """
        if iftrain == False:
            N = 3
            weight = torch.zeros((x.size(0), 513))
            std = 30
            disturb = torch.distributions.normal.Normal(0, std)

            for mm in range(1, N+1):
                noise = disturb.sample(x.size()).to(device=x.device)
                xm = x + noise
                mu, logvar, y4, _ = self.encode(xm)
                weight = weight + self.in_score(mu, y4)
            weight = weight / N
            mu, logvar, y4, y = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar, y4, weight, y

        else:
            mu, logvar, y4, y = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar, z, y


class ImageDiscriminator(nn.Module):
    """
    Discriminator operating on reconstructed images.
    Used for adversarial training of the SeAugVAE model.
    """
    def __init__(self):
        super(ImageDiscriminator, self).__init__()

        self.en = resnet18()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        z, y1, y2, y3, y4 = self.en(x)
        z = self.fc(z)
        return z, y4

class FeatureDiscriminator(nn.Module):
    """
    Discriminator operating in latent space.
    Used to distinguish true latent features from generated ones.
    """
    def __init__(self):
        super(FeatureDiscriminator, self).__init__()

        self.fcc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        z = self.fcc(x)
        return z



