# Multimodal Deepfake Detection
## Project Documentation

---

## 1. Overview

An explainable, multimodal deepfake detection system that classifies social-media videos into six categories:

| Class | Description |
|---|---|
| `real` | Authentic, unmanipulated video |
| `ai_video` | Fully AI-generated / DeepFake video |
| `faceswap` | Face-swap manipulation |
| `edited` | Face2Face / NeuralTextures expression manipulation |
| `voiceclone` | Synthetic / cloned voice audio |
| `audio_video_fake` | Combined audio-visual manipulation |

The system provides a per-video forensic report (decision, category, confidence, risk level, supporting evidence) via both a command-line tool and a Streamlit web dashboard.

---

## 2. Architecture

```
Input Video
    ↓
Frame Sampling (every Nth frame)
    ↓
YOLOv8 Face Detection + Cropping
    ↓
ResNet18 Classifier (6-class, fine-tuned)
    ↓
Per-frame Softmax Prediction
    ↓
Majority Vote Across Sampled Frames
    ↓
Explainable Forensic Report
```

The visual pipeline (ResNet18) is fully implemented and trained. A parallel research-track architecture (EfficientNet-B0 + ViT visual branch, Wav2Vec2 audio branch, learnable fusion network) was also built; the audio branch and fusion network are trained but **not currently wired into the production inference path** (`detect_social_video.py` / Streamlit app run visual-only). This is listed as future work in Section 6.

---

## 3. Dataset

| Class | Source | Videos | Extracted Faces |
|---|---|---|---|
| real | FF++ originals + CelebDF real | — | 6,186 |
| ai_video | FF++ DeepFakes + CelebDF fake | — | 7,711 |
| faceswap | FF++ FaceSwap (c40) | 150 | 2,293 |
| edited | FF++ Face2Face + NeuralTextures (c40) | 100 | 1,469 |
| voiceclone | FakeAVCeleb (FARV) | — | 253 |
| audio_video_fake | FakeAVCeleb (RAFV + FAFV) | — | 674 |

**Total: 18,586 face crops**, split 80/20 into train (14,869) and validation (3,717) sets, stratified by class via `WeightedRandomSampler` on the training split only.

---

## 4. Training Configuration

- **Backbone:** ResNet18 (ImageNet pretrained), final layer replaced with 6-class output
- **Augmentation (train only):** horizontal flip, ±10° rotation, color jitter
- **Class imbalance handling:** `WeightedRandomSampler` (inverse class frequency) on training data
- **Optimizer:** Adam, lr=1e-4
- **Epochs:** 15
- **Checkpointing:** best model saved by lowest validation loss (not training loss)

---

## 5. Results

### Final validation metrics (18,586-image dataset, post data-augmentation of faceswap/edited classes)

```
                  precision    recall  f1-score   support

        ai_video       0.99      0.99      0.99      1571
audio_video_fake       1.00      1.00      1.00       118
          edited       0.87      0.80      0.83       313
        faceswap       0.95      0.97      0.96       460
            real       0.94      0.92      0.93      1202
      voiceclone       0.49      0.89      0.63        53

        accuracy                           0.95      3717
       macro avg       0.87      0.93      0.89      3717
    weighted avg       0.95      0.95      0.95      3717
```

**Overall validation accuracy: 95%**

`voiceclone` remains the weakest class (49% precision) due to its small sample count (253 faces total) relative to other classes — a direct product of limited voice-clone footage available in the source dataset. This is a known, tracked weakness (see Section 6).

---

## 6. Known Limitation: Cross-Dataset Generalization Gap

### 6.1 The problem

The model was validated at 95% accuracy **on held-out data from the same source distribution it was trained on** — FaceForensics++ (research-grade, c40-compressed, medium-shot interview-style footage) and CelebDF/FakeAVCeleb.

When tested on **real-world, in-the-wild videos** — specifically two independently sourced test videos, one AI-generated deepfake and one manipulated WhatsApp-shared video, both in vertical mobile format (720×1280) with close-up selfie-style framing — the model incorrectly classified both as `real` with 80–88% confidence, despite both being confirmed manipulated content.

### 6.2 Root cause

This is a **distribution shift** problem, not a code defect. The training data (FF++, CelebDF) consists almost entirely of:
- Landscape-oriented, medium-shot footage
- Source videos originally scraped from YouTube (interview/broadcast style)
- Specific, uniform compression profiles (c40)

Real-world social media video differs systematically:
- Portrait/vertical orientation, common on Instagram/WhatsApp/TikTok
- Extreme close-up, front-camera selfie framing
- Platform-specific compression (WhatsApp, Instagram re-encoding) distinct from FF++'s c40 profile

A CNN classifier trained exclusively on one visual distribution does not automatically generalize to another, even when the semantic task (real vs. fake face) is identical. This is a well-documented phenomenon in deepfake detection literature — models trained on FaceForensics++ commonly show significant accuracy drops when evaluated on in-the-wild benchmarks such as DFDC (DeepFake Detection Challenge) or Celeb-DF's "in-the-wild" split.

### 6.3 Why this doesn't invalidate the 95% result

The 95% validation accuracy is a valid, honestly-measured result **for the distribution it was measured on**. It demonstrates the architecture, class-imbalance handling, and training pipeline all function correctly. The gap identified here is specifically a **generalization boundary**, not a training or evaluation error — the two are separate claims and this document keeps them distinct rather than letting one undermine the other.

### 6.4 Planned fix (tracked separately, ~1 week effort)

- Incorporate the **DFDC (DeepFake Detection Challenge)** dataset, which contains diverse, in-the-wild-style footage (varied cameras, compression, lighting, orientation) explicitly designed to stress-test generalization
- Re-run the full pipeline: frame extraction → YOLO face cropping → dataset merge → retraining
- Re-evaluate on the same two real-world test videos used in this report to confirm the fix, plus ideally a held-out third video not used during any prior iteration

---

## 7. Component Status

| Component | Status |
|---|---|
| Dataset integration (FF++, CelebDF, FakeAVCeleb) | ✅ Complete |
| Frame extraction (all 6 classes) | ✅ Complete |
| YOLOv8 face cropping (all 6 classes) | ✅ Complete |
| 6-class ResNet18 training with val split | ✅ Complete |
| Class imbalance handling (WeightedRandomSampler) | ✅ Complete |
| Confusion matrix / classification report evaluation | ✅ Complete |
| CLI forensic report (`detect_social_video.py`) | ✅ Complete — real inference, no mock values |
| Streamlit dashboard | ✅ Complete — real inference, connected to trained model |
| EfficientNet + ViT visual branch | ✅ Built, trained, not wired into production inference |
| Wav2Vec2 audio branch | ✅ Built, trained, not wired into production inference |
| Learnable fusion model | ✅ Built, trained, not wired into production inference |
| Cross-dataset generalization (DFDC) | ⏸️ Planned, not started |

---

## 8. Reproducing Results

```powershell
# Train
python -m training.train_video

# CLI inference
python detect_social_video.py <video_file.mp4>

# Dashboard
streamlit run app\streamlit_app.py
```