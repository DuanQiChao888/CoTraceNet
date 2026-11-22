"""Independent reconstruction of ShuffleBANet, Li et al., TDSC 2025.

Based on Figures 3–6, Tables I and IV, Algorithm 1 and Equations 1–4
(DOI 10.1109/TDSC.2025.3596710). Ambiguous choices are identified below.
Input is a batch of floating-point, single-channel images.
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(in_channels, out_channels, kernel=1, stride=1, depthwise=False, activation=None):
    groups = in_channels if depthwise else 1
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


def shuffle_channels(features):
    """Interleave two equal channel groups. The paper omits the group count."""
    batch, channels, height, width = features.shape
    groups = features.reshape(batch, 2, channels // 2, height, width)
    return groups.transpose(1, 2).reshape(batch, channels, height, width)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = max(1, channels // 16)  # Section III-C gives r=16.
        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return features * self.weights(features)


class CoordinateAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = max(8, channels // 32)  # CA bottleneck width is unspecified.
        self.compress = conv_bn(channels, reduced)
        self.row_gate = nn.Conv2d(reduced, channels, 1)
        self.col_gate = nn.Conv2d(reduced, channels, 1)

    def forward(self, features):
        height, width = features.shape[-2:]
        rows = features.mean(dim=3, keepdim=True)
        cols = features.mean(dim=2, keepdim=True).transpose(2, 3)
        joined = self.compress(torch.cat((rows, cols), dim=2))
        rows, cols = joined.split((height, width), dim=2)
        return (
            features * self.row_gate(rows).sigmoid() * self.col_gate(cols.transpose(2, 3)).sigmoid()
        )


class EnhancedPreprocessing(nn.Module):
    def __init__(self, srm_kernels):
        super().__init__()
        filters = torch.as_tensor(srm_kernels, dtype=torch.float32)
        if filters.shape != (30, 1, 5, 5):
            raise ValueError("srm_kernels must have shape (30, 1, 5, 5)")
        # The paper specifies 30 SRM filters but not whether they are updated.
        self.register_buffer("srm_kernels", filters.detach().clone())
        self.normalize = nn.Sequential(nn.BatchNorm2d(30), nn.GELU())
        # Figure 4b shows a skip connection. Internal MBConv expansion is
        # unspecified. Here it is 3, as in the adjacent ShuffleBottleneck.
        self.mobile = nn.Sequential(
            conv_bn(30, 90, activation=nn.SiLU()),
            conv_bn(90, 90, 3, depthwise=True, activation=nn.SiLU()),
            ChannelAttention(90),
            conv_bn(90, 30),
        )
        self.bottleneck = nn.Sequential(
            conv_bn(30, 90, activation=nn.ReLU()),
            conv_bn(90, 90, 3, depthwise=True, activation=nn.ReLU()),
            conv_bn(90, 30),
        )

    def forward(self, images):
        # Valid 5x5 convolution gives 252x124 from 256x128. Table IV's
        # 252x126 is inconsistent with symmetric valid 5x5 filtering.
        edges = self.normalize(F.conv2d(images, self.srm_kernels))
        # Table IV preserves this resolution through preprocessing. Table I's
        # stride of 2 conflicts with its own stated unchanged output size.
        return shuffle_channels(self.bottleneck(edges + self.mobile(edges)))


class ShuffleBA1(nn.Module):
    """Figure 5a. Both full-input paths downsample, then merge."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        half = out_channels // 2
        self.local = nn.Sequential(
            conv_bn(in_channels, in_channels, 3, 2, depthwise=True),
            ChannelAttention(in_channels),
            conv_bn(in_channels, half, activation=nn.GELU()),
        )
        self.wide = nn.Sequential(
            conv_bn(in_channels, in_channels, 7, 2, depthwise=True, activation=nn.GELU()),
            conv_bn(in_channels, half),
            conv_bn(half, half, activation=nn.GELU()),
        )

    def forward(self, features):
        both = torch.cat((self.local(features), self.wide(features)), dim=1)
        return shuffle_channels(both)


class ShuffleBA2(nn.Module):
    """Figure 5b. Keep half the channels and transform the other half."""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        # Section III-C specifies expansion/reduction but not the internal
        # width. Use the full stage width (2x the transformed half).
        self.transform = nn.Sequential(
            conv_bn(half, channels, activation=nn.GELU()),
            conv_bn(channels, channels, 3, depthwise=True),
            ChannelAttention(channels),
            conv_bn(channels, half, activation=nn.GELU()),
        )

    def forward(self, features):
        kept, changed = features.chunk(2, dim=1)
        return shuffle_channels(torch.cat((kept, self.transform(changed)), dim=1))


class ShuffleBANet(nn.Module):
    def __init__(self, srm_kernels):
        super().__init__()
        layers = [EnhancedPreprocessing(srm_kernels)]
        previous = 30
        for width, repeats in ((64, 3), (128, 8), (256, 3)):
            layers.append(ShuffleBA1(previous, width))
            layers.extend(ShuffleBA2(width) for _ in range(repeats))
            previous = width
        layers.extend(
            [CoordinateAttention(256), conv_bn(256, 512), nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        )
        self.encoder = nn.Sequential(*layers)
        # Section IV gives dropout 0.5 without specifying its placement.
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(4 * 512, 2))

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected monochrome images of shape (B, 1, H, W)")
        height, width = images.shape[-2:]
        if height < 5 or width < 10 or width % 2:
            raise ValueError("Expected H >= 5 and even W >= 10 for equal halves")
        left_image, right_image = images.chunk(2, dim=3)
        # Use a common batch for both halves and a single set of BN statistics.
        both = self.encoder(torch.cat((left_image, right_image), dim=0))
        left, right = both.chunk(2, dim=0)
        pair = torch.stack((left, right), dim=1)
        # Equation 1 omits the denominator. The cited SiaStegNet reference
        # uses the sample variance across the two halves.
        fusion = torch.cat(
            (pair.amax(1), pair.amin(1), pair.mean(1), pair.var(1, unbiased=True)), dim=1
        )
        return self.classifier(fusion), left, right

    def forward(self, images):
        logits, _, _ = self.forward_features(images)
        return logits


def shuffle_loss(logits, left, right, labels, alpha=0.1, margin=1.25):
    """Equations 2–4, label 0 = cover, 1 = stego. Inputs are raw logits."""
    distance = torch.linalg.vector_norm(left - right, dim=1)
    y = labels.to(dtype=distance.dtype)
    contrastive = ((1 - y) * distance.square() + y * F.relu(margin - distance).square()) / 2
    return F.cross_entropy(logits, labels) + alpha * contrastive.mean()
