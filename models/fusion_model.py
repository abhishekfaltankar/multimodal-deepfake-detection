import torch
import torch.nn as nn

class FusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(6 + 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 6)
        )

    def forward(self, video_logits, audio_logits):

        x = torch.cat(
            [video_logits, audio_logits],
            dim=1
        )

        return self.fusion(x)