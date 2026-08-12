import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTModel

class VisualExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        cnn = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.cnn = nn.Sequential(*list(cnn.children())[:-1])

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

        self.fc = nn.Linear(1280 + 768, 512)

    def forward(self, image, vit_pixels):

        cnn_feat = self.cnn(image)
        cnn_feat = cnn_feat.flatten(1)

        vit_out = self.vit(pixel_values=vit_pixels)
        vit_feat = vit_out.last_hidden_state[:, 0]

        fused = torch.cat([cnn_feat, vit_feat], dim=1)

        return self.fc(fused)