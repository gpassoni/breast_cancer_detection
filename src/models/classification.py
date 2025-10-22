"""Classification model using pretrained backbone."""
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for breast cancer detection."""
    
    def __init__(self, backbone, backbone_output_dim=2048):
        super(ClassificationHead, self).__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
