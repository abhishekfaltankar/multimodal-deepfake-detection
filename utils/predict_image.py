import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# CNN architecture
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Load trained model
model = DeepfakeCNN()
model.load_state_dict(torch.load("saved_models/deepfake_cnn.pth", map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Automatically select first image
image_path = next(Path("dataset/prepared/real").glob("*.jpg"))
print("Testing image:", image_path)

# Load image
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()

classes = ["fake", "real"]
print("Prediction:", classes[prediction])