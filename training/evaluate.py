import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.video_model import DeepfakeResNet18

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(
    "processed/faces",
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = DeepfakeResNet18()
model.load_state_dict(torch.load(
    "saved_models/social_media_detector.pth",
    map_location=device
))
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Evaluation images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")