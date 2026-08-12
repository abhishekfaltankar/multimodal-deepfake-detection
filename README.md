# Multimodal Deepfake Detection

An explainable, multimodal deepfake detection system that classifies social-media videos into six categories: **real**, **AI-generated video**, **faceswap**, **edited (Face2Face/NeuralTextures)**, **voiceclone**, and **audio-video fake**. The system provides a per-video forensic report — decision, category, confidence, risk level, and supporting evidence — through both a command-line tool and a Streamlit web dashboard.

📄 **For full technical details — architecture, dataset stats, training configuration, evaluation results, and known limitations — see [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md).**

---

## Features

- YOLOv8-based face detection and cropping
- Fine-tuned ResNet18 classifier (6-class)
- Class-imbalance handling via `WeightedRandomSampler`
- Explainable forensic reports with per-class visual/audio evidence
- Command-line inference tool
- Streamlit web dashboard with downloadable reports

## Project Status

- ✅ Core visual detection pipeline: trained, evaluated (95% validation accuracy), and deployed
- ✅ CLI and Streamlit dashboard: fully connected to the trained model
- ⏸️ Known limitation: reduced accuracy on real-world/in-the-wild mobile video (see [Section 6, PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#6-known-limitation-cross-dataset-generalization-gap)) — DFDC dataset integration planned to address this
- ⏸️ Audio branch (Wav2Vec2) and fusion model: trained but not yet wired into production inference

## Setup

```powershell
git clone https://github.com/multimodal-deepfake-team/multimodal-deepfake-detection.git
cd multimodal-deepfake-detection

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install torch torchvision torchaudio ultralytics streamlit scikit-learn transformers opencv-python
```

**Note:** Dataset files, trained model weights, and YOLO weights are not included in this repository (excluded via `.gitignore` due to size and licensing). Contact a team member for access to `saved_models/social_media_detector.pth` and `weights/yolov8n-face.pt`, or follow the dataset setup steps in [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md#3-dataset) to rebuild them from source.

## Usage

**Train the model:**
```powershell
python -m training.train_video
```

**Run inference on a video (CLI):**
```powershell
python detect_social_video.py <video_file.mp4>
```

**Launch the web dashboard:**
```powershell
streamlit run app\streamlit_app.py
```

## Project Structure

```
multimodal-deepfake-detection/
├── app/                  # Streamlit dashboard
├── models/               # Model architecture definitions
├── training/             # Training scripts
├── preprocessing/        # Frame extraction, face cropping
├── detect_social_video.py  # CLI inference tool
├── PROJECT_DOCUMENTATION.md  # Full technical report
└── README.md
```

## Team

Maintained by the Multimodal Deepfake Detection team.