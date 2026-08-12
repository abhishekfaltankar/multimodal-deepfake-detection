import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# Load model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(
    torch.load("saved_models/deepfake_resnet18.pth", map_location="cpu")
)

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Automatically select the first fake image
fake_dir = Path("dataset/ffpp_prepared/fake")

image_path = "dataset/ffpp_prepared/fake/frame_0010_face0.jpg"

print(f"Testing image: {image_path}")

# Load image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

classes = ['fake', 'real']

# Predict
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    probabilities = torch.softmax(output, dim=1)[0]

print(f"Predicted class index: {prediction}")
print(f"Fake probability: {probabilities[0]:.4f}")
print(f"Real probability: {probabilities[1]:.4f}")
print(f"Prediction: {classes[prediction]}")