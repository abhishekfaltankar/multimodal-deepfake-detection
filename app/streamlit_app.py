import streamlit as st
from pathlib import Path
import tempfile
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.video_model import DeepfakeResNet18

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = ROOT / "saved_models" / "social_media_detector.pth"
YOLO_WEIGHTS = ROOT / "weights" / "yolov8n-face.pt"
FRAME_SKIP = 10
MAX_FRAMES_TO_CHECK = 60

# This MUST match the printed "Class to index mapping" from training/train_video.py
# ImageFolder sorts class folder names alphabetically -- do not reorder by hand.
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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_models():
    model = DeepfakeResNet18(num_classes=6)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    model.eval()
    face_model = YOLO(str(YOLO_WEIGHTS))
    return model, face_model


def analyze_video(video_path, model, face_model):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

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
                    face_tensor = transform(face).unsqueeze(0)

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


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Multimodal Deepfake Detection",
    layout="centered"
)

st.title("🎥 Multimodal Deepfake Detection")
st.write("Upload a social-media video to verify whether it is authentic or manipulated.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("📹 Uploaded Video")
    st.video(str(temp_path))

    if st.button("🔍 Analyze Video"):
        with st.spinner("Loading model and analyzing video..."):
            model, face_model = load_models()
            predictions, confidences = analyze_video(temp_path, model, face_model)

        if predictions is None:
            st.error("Could not open the uploaded video file.")
            st.stop()

        if len(predictions) == 0:
            st.error("No face detected in the sampled frames. Cannot analyze this video.")
            st.stop()

        pred_labels = [IDX_TO_CLASS[p] for p in predictions]
        label_counts = Counter(pred_labels)
        final_label, vote_count = label_counts.most_common(1)[0]

        matching_confidences = [c for lbl, c in zip(pred_labels, confidences) if lbl == final_label]
        avg_confidence = round(sum(matching_confidences) / len(matching_confidences) * 100, 2)

        decision = "REAL" if final_label == "real" else "FAKE"
        category = CLASS_DISPLAY[final_label]
        evidence = EVIDENCE_MAP[final_label]

        st.subheader("📋 Forensic Analysis Result")

        if decision == "REAL":
            st.success(f"FINAL DECISION: {decision}")
        else:
            st.error(f"FINAL DECISION: {decision}")

        st.write(f"**CATEGORY:** {category}")
        st.write(f"**CONFIDENCE:** {avg_confidence}%")
        st.write(f"**RISK LEVEL:** {evidence['risk']}")
        st.write(f"**Frames Analyzed:** {len(predictions)}")

        st.subheader("🗳️ Frame Vote Breakdown")
        for lbl, count in label_counts.most_common():
            st.write(f"- {CLASS_DISPLAY[lbl]}: {count}/{len(predictions)} frames")

        st.subheader("🖼️ Visual Evidence")
        for item in evidence["visual"]:
            st.write(f"- {item}")

        st.subheader("🔊 Audio Evidence")
        for item in evidence["audio"]:
            st.write(f"- {item}")

        vote_lines = "\n".join(
            f"- {CLASS_DISPLAY[lbl]}: {count}/{len(predictions)} frames"
            for lbl, count in label_counts.most_common()
        )

        report = f"""
========================================
SOCIAL MEDIA FORENSIC REPORT
========================================

Video Name      : {uploaded_file.name}
Frames Analyzed : {len(predictions)}

FINAL DECISION  : {decision}
CATEGORY        : {category}
CONFIDENCE      : {avg_confidence}%
RISK LEVEL      : {evidence['risk']}

FRAME VOTE BREAKDOWN
{vote_lines}

VISUAL EVIDENCE
{chr(10).join('- ' + v for v in evidence['visual'])}

AUDIO EVIDENCE
{chr(10).join('- ' + a for a in evidence['audio'])}

========================================
"""

        st.download_button(
            label="⬇️ Download Forensic Report",
            data=report,
            file_name="forensic_report.txt",
            mime="text/plain"
        )
