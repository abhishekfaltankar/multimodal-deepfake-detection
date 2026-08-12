from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter

from models.video_model import DeepfakeResNet18

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "saved_models/social_media_detector.pth"
YOLO_WEIGHTS = "weights/yolov8n-face.pt"
FRAME_SKIP = 10          # process every Nth frame (match your extract_frames.py interval)
MAX_FRAMES_TO_CHECK = 60  # cap how many frames we run inference on, for speed

# This MUST match the printed "Class to index mapping" from training/train_video.py
# ImageFolder sorts class folder names alphabetically -- do not reorder this by hand.
IDX_TO_CLASS = {
    0: "ai_video",
    1: "audio_video_fake",
    2: "edited",
    3: "faceswap",
    4: "real",
    5: "voiceclone",
}

CLASS_DISPLAY = {
    "real": "REAL",
    "ai_video": "AI_VIDEO",
    "faceswap": "FACESWAP",
    "edited": "EDITED",
    "voiceclone": "VOICECLONE",
    "audio_video_fake": "AUDIO_VIDEO_FAKE",
}

EVIDENCE_MAP = {
    "real": {
        "visual": ["No visual manipulation detected"],
        "audio": ["Audio appears authentic"],
        "risk": "LOW",
    },
    "ai_video": {
        "visual": ["Face boundary inconsistency", "Temporal flicker detected", "Compression artifact mismatch"],
        "audio": ["No suspicious audio detected"],
        "risk": "HIGH",
    },
    "faceswap": {
        "visual": ["Face blending seam detected", "Identity mismatch across frames", "Unnatural face-background boundary"],
        "audio": ["No suspicious audio detected"],
        "risk": "HIGH",
    },
    "edited": {
        "visual": ["Localized facial region manipulation", "Expression-motion inconsistency", "Neural texture artifacts"],
        "audio": ["No suspicious audio detected"],
        "risk": "MEDIUM",
    },
    "voiceclone": {
        "visual": ["No visual manipulation detected"],
        "audio": ["Synthetic speech pattern detected", "Voice-cloning artifacts in spectral features"],
        "risk": "HIGH",
    },
    "audio_video_fake": {
        "visual": ["Face boundary inconsistency", "Temporal flicker detected"],
        "audio": ["Synthetic speech pattern detected", "Audio-visual sync anomaly"],
        "risk": "HIGH",
    },
}

# ---------------------------
# Setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = DeepfakeResNet18(num_classes=6).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_and_predict(video_path, model, face_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        sys.exit(1)

    predictions = []
    confidences = []

    frame_idx = 0
    checked_frames = 0

    while checked_frames < MAX_FRAMES_TO_CHECK:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            results = face_model(frame, verbose=False)

            if len(results[0].boxes) > 0:
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    face_tensor = transform(face).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = F.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)

                    predictions.append(pred_idx.item())
                    confidences.append(conf.item())
                    checked_frames += 1

        frame_idx += 1

    cap.release()
    return predictions, confidences


def build_report(video_path, predictions, confidences):
    if len(predictions) == 0:
        print()
        print("========================================")
        print("SOCIAL MEDIA FORENSIC REPORT")
        print("========================================")
        print()
        print(f"Video Name      : {video_path.name}")
        print()
        print("FINAL DECISION  : UNKNOWN")
        print("REASON          : No face detected in sampled frames")
        print("========================================")
        return

    pred_labels = [IDX_TO_CLASS[p] for p in predictions]
    label_counts = Counter(pred_labels)

    # majority vote across sampled frames
    final_label, vote_count = label_counts.most_common(1)[0]

    # average confidence only over frames that voted for the winning class
    matching_confidences = [c for lbl, c in zip(pred_labels, confidences) if lbl == final_label]
    avg_confidence = round(sum(matching_confidences) / len(matching_confidences) * 100, 2)

    decision = "REAL" if final_label == "real" else "FAKE"
    category = CLASS_DISPLAY[final_label]
    evidence = EVIDENCE_MAP[final_label]

    print()
    print("========================================")
    print("SOCIAL MEDIA FORENSIC REPORT")
    print("========================================")
    print()
    print(f"Video Name      : {video_path.name}")
    print(f"Frames Analyzed : {len(predictions)}")
    print()
    print(f"FINAL DECISION  : {decision}")
    print(f"CATEGORY        : {category}")
    print(f"CONFIDENCE      : {avg_confidence}%")
    print(f"RISK LEVEL      : {evidence['risk']}")
    print()
    print(f"FRAME VOTE BREAKDOWN")
    for lbl, count in label_counts.most_common():
        print(f"- {CLASS_DISPLAY[lbl]}: {count}/{len(predictions)} frames")
    print()
    print("VISUAL EVIDENCE")
    for item in evidence["visual"]:
        print(f"- {item}")
    print()
    print("AUDIO EVIDENCE")
    for item in evidence["audio"]:
        print(f"- {item}")
    print()
    print("========================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_social_video.py <video_file>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    print("Loading model...")
    model = load_model()
    face_model = YOLO(YOLO_WEIGHTS)

    print("Analyzing video...")
    predictions, confidences = extract_and_predict(video_path, model, face_model)

    build_report(video_path, predictions, confidences)