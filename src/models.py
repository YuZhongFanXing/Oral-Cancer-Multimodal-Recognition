"""
Core model components for multimodal oral cancer recognition.

Architecture:
  EfficientNetV2-S (1280d) -> SEBlock(1280) -> Concat
  MetaMLP(5->128->256) ----/
  FC(1536->1024->512->256->2)
"""

import torch
import torch.nn as nn
from torchvision import models


def get_efficientnet_v2s():
    """EfficientNetV2-S with ImageNet pretrained weights, classifier removed."""
    m = models.efficientnet_v2_s(weights='DEFAULT')
    dim = m.classifier[1].in_features  # 1280
    m.classifier = nn.Identity()
    return m, dim


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for 1D feature vectors.

    Applied after global pooling to recalibrate channel responses.
    x_out = x * sigmoid(FC(ReLU(FC(x))))
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MetaMLP(nn.Module):
    """Encodes 5-dim clinical metadata (Age, Gender, Smoking, Betel, Alcohol)
    into a 256-dim feature vector.

    Architecture: 5 -> 128 -> 256 with BatchNorm + SiLU + Dropout.
    """
    def __init__(self, meta_dim=5, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),    nn.Dropout(0.2),
            nn.Linear(128, out_dim),  nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True),    nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)


def build_head(in_dim, num_classes=2):
    """Classifier head: in -> 1024 -> 512 -> 256 -> 2."""
    return nn.Sequential(
        nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),   nn.Dropout(0.5),
        nn.Linear(1024, 512),    nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),   nn.Dropout(0.4),
        nn.Linear(512, 256),     nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),   nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )


class ImgSEModel(nn.Module):
    """Full multimodal model with SE channel attention.

    Forward:
      1. backbone(img)              -> 1280d image features
      2. img_se(img_feat)           -> 1280d (channel-recalibrated)
      3. meta_branch(meta)          -> 256d metadata features
      4. cat([img_feat, meta_feat]) -> 1536d fused features
      5. classifier(fused)          -> 2-class logits
    """
    def __init__(self, meta_dim=5, num_classes=2):
        super().__init__()
        self.backbone, img_out = get_efficientnet_v2s()       # 1280d
        self.img_se = SEBlock(img_out, reduction=16)           # SE(1280, 16)
        self.meta_branch = MetaMLP(meta_dim, out_dim=256)
        fusion_dim = img_out + 256                             # 1536d
        self.classifier = build_head(fusion_dim, num_classes)

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        img_feat = self.img_se(img_feat)
        meta_feat = self.meta_branch(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)
