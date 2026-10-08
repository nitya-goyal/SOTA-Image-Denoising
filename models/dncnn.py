"""
DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (IEEE TIP 2017)
Paper: Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, Lei Zhang
ArXiv: https://arxiv.org/abs/1608.03981
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN: Deep Residual Learning CNN for Gaussian Image Denoising.
    Learns to predict the residual noise R(y) = y - x, where y is noisy image and x is clean image.
    """
    def __init__(self, in_channels=3, out_channels=3, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: Conv (predicts residual noise map)
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Predict noise residual
        residual = self.dncnn(x)
        # Reconstruct clean image: clean = noisy - residual
        return x - residual
