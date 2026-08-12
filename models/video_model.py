import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DeepfakeResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Load pretrained ResNet18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)