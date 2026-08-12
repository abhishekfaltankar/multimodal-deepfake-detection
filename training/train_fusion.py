import torch

from models.fusion_model import FusionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = FusionModel().to(device)

dummy_video = torch.randn(8, 6).to(device)

dummy_audio = torch.randn(8, 3).to(device)

output = model(dummy_video, dummy_audio)

print("Fusion output shape:", output.shape)

torch.save(
    model.state_dict(),
    "saved_models/fusion_model.pth"
)

print("Fusion model saved successfully")