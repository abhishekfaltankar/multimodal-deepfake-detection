from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from models.video_model import DeepfakeResNet18

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "processed/faces"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42
SAVE_PATH = "saved_models/social_media_detector.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Transforms
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# Dataset + split
# ---------------------------
base_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform, allow_empty=True)
print(f"Total images: {len(base_dataset)}")
print(f"Classes: {base_dataset.classes}")
print(f"Class to index mapping: {base_dataset.class_to_idx}")

targets = base_dataset.targets
print(f"Samples per class: {dict(sorted(Counter(targets).items()))}")

num_val = int(len(base_dataset) * VAL_SPLIT)
num_train = len(base_dataset) - num_val

generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(base_dataset, [num_train, num_val], generator=generator)

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self.samples = [base_dataset.samples[i] for i in subset.indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = base_dataset.loader(path)
        img = self.transform(img)
        return img, label

train_dataset = TransformedSubset(train_subset, train_transform)
val_dataset = TransformedSubset(val_subset, val_transform)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

# ---------------------------
# WeightedRandomSampler (train set only)
# ---------------------------
train_targets = [label for _, label in train_dataset.samples]
class_sample_counts = Counter(train_targets)
print(f"Train samples per class: {dict(sorted(class_sample_counts.items()))}")

num_classes = len(base_dataset.classes)
class_weights = [0.0] * num_classes
for cls_idx, count in class_sample_counts.items():
    class_weights[cls_idx] = 1.0 / count

sample_weights = [class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model
# ---------------------------
model = DeepfakeResNet18(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def run_validation(model, loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return val_loss / total, 100.0 * correct / total, all_preds, all_labels


# ---------------------------
# Training loop
# ---------------------------
best_val_loss = float("inf")
Path("saved_models").mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)}", flush=True)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    val_loss, val_acc, _, _ = run_validation(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> Best model saved (val loss {val_loss:.4f})")

print("\nTraining complete.")

# ---------------------------
# Final evaluation with best model
# ---------------------------
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

_, _, all_preds, all_labels = run_validation(model, val_loader)

print("\nConfusion Matrix (rows=true, cols=predicted):")
print(f"Class order: {base_dataset.classes}")
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
print(cm)

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    labels=list(range(num_classes)),
    target_names=base_dataset.classes,
    zero_division=0
))