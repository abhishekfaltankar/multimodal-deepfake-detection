---
title: Deepfake Detector
emoji: 🎙️
sdk: docker
app_port: 7860
---

# Deepfake Detector (voice + face)

Upload a video and get **REAL / FAKE / UNCERTAIN**. The **voice** and the **face** are checked
by two separate pretrained models (no training), then combined into one verdict.

| Part | Model | How it works |
|------|-------|--------------|
| Voice | `Speech-Arena-2025/DF_Arena_500M_V_1` | ffmpeg extracts audio -> silence removed -> 4 s windows -> each window scored -> averaged |
| Face | `HoopitAI/video-deepfake-detection-GenD_DINOv3_L_FF` | ~24 frames sampled -> largest face cropped -> each crop scored -> averaged |

If a video has no face (or no speech) the other part is used on its own, and the app says so.
A single very confident part (>= 0.90 fake) is enough to flag the video as FAKE.

## Files
- `app.py`: Streamlit web app
- `detect.py`: voice part + fusion (settings at the top: models, thresholds, weights)
- `visual.py`: face part
- `evaluate.py`: accuracy, precision, recall, F1, ROC-AUC, confusion matrix on your own videos
- `Dockerfile`: for deploying on Hugging Face Spaces

## Run locally
1. Python 3.11 virtual environment (ffmpeg comes bundled through `imageio-ffmpeg`, no separate install needed)
2. Then:
```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Before starting the app, run `python check_setup.py`. It tests Python, packages, ffmpeg and both
models separately and tells you exactly what to fix.
The first run downloads both models (a few GB), so it is slow once.
Tip (Windows): keep the project in a short path like `C:\dd` (not inside OneDrive), because
very long paths break package installs.
Quick command-line test: `python detect.py clip.mp4`

If the face model asks for access to a gated model (for example facebook/dinov3), accept its
license on Hugging Face and run `huggingface-cli login`. If an import error names a missing
package, `pip install` it.

## Check accuracy on your own videos (do this before trusting it)
```bash
pip install scikit-learn matplotlib
# put videos in dataset/real/ and dataset/fake/ (aim for ~100 total, include reels)
python evaluate.py
```
Report the **HOLDOUT** block. It also prints the ROC-AUC of the voice part and the face part
separately: give more weight (`WEIGHTS` in `detect.py`) to the part that scores better, and
copy the suggested thresholds into `detect.py`.

## Upload to GitHub
```bash
git init
git add .
git commit -m "Deepfake detector"
git branch -M main
git remote add origin https://github.com/<your-username>/deepfake-detector.git
git push -u origin main
```
Your test videos in `dataset/` are not uploaded (see `.gitignore`).

## Deploy on Hugging Face Spaces (Docker), recommended
Streamlit Community Cloud has too little memory for these models. Hugging Face Spaces gives
16 GB RAM on CPU Basic hardware. Check Hugging Face's current rules for free compute Spaces.
1. Create a Hugging Face account and a token (Settings -> Access Tokens, role: write)
2. New Space -> SDK: **Docker** -> Blank -> create
3. Push this repo to it (the password is your token):
```bash
git remote add space https://huggingface.co/spaces/<your-username>/deepfake-detector
git push space main
```
The Space sleeps when unused and wakes on the next visit (the first load is slow).
To start faster, uncomment the model pre-download line in the `Dockerfile`.

## Limits
- The face model was trained on FaceForensics++ (older face-swap methods). Its published accuracy
  is on that same dataset, so expect lower accuracy on brand-new AI-generated video.
- Face detection is a simple frontal-face detector: side views and tiny faces are skipped.
- Background music, heavy compression and new voice generators can reduce voice accuracy.
- Results are probabilities, not proof.
