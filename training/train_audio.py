from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from models.audio_extractor import AudioExtractor

CLASSES = {
    "real": 0,
    "voiceclone": 1,
    "audio_video_fake": 2
}

class AudioDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        root = Path(root)

        for cls_name, label in CLASSES.items():

            folder = root / cls_name

            if not folder.exists():
                continue

            for wav in folder.glob("*.wav"):
                self.samples.append((wav, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        wav_path, label = self.samples[idx]

        waveform, sr = torchaudio.load(wav_path)

        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform, sr, 16000
            )

        waveform = waveform.mean(dim=0)

        waveform = waveform[:16000]

        if waveform.shape[0] < 16000:
            pad = 16000 - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform, label


dataset = AudioDataset("processed/audio")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioExtractor().to(device)

classifier = nn.Linear(256, 3).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=1e-4
)

EPOCHS = 5

print(f"Audio samples: {len(dataset)}")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for audio, labels in loader:

        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        features = model(audio)

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}"
    )

torch.save(
    {
        "extractor": model.state_dict(),
        "classifier": classifier.state_dict()
    },
    "saved_models/audio_classifier.pth"
)

print("Audio model saved successfully")