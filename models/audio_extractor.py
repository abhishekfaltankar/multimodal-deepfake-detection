import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.proj = nn.Linear(768, 256)

    def forward(self, audio):

        out = self.wav2vec(audio)

        feat = out.last_hidden_state.mean(dim=1)

        return self.proj(feat)