# 🛡️ Multi-Modal Deepfake Detection Dashboard ✨

**An AI-powered deepfake detection system that analyzes both video and audio to identify potentially manipulated media.** The system evaluates facial patterns and voice characteristics through separate analytical pipelines, then presents individual scores and a combined verdict through an interactive Streamlit dashboard.

> 🎯 **Project Goal:** Make deepfake analysis more accessible by combining visual and acoustic evidence in one interface.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Deep Learning Models](#-deep-learning-models)
- [Technology Stack](#-technology-stack)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Cloud Deployment](#-cloud-deployment)
- [Report and Results](#-report-and-results)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview

Deepfakes use artificial intelligence to generate or manipulate faces, videos, and voices. These technologies can be used creatively, but they can also enable misinformation, impersonation, and digital fraud.

The **Multi-Modal Deepfake Detection Dashboard** is designed to analyze two important sources of evidence:

- 🎥 **Video analysis:** Evaluates facial information extracted from video frames.
- 🎙️ **Audio analysis:** Evaluates voice characteristics to identify patterns associated with synthetic or manipulated speech.
- 🧠 **Combined analysis:** Uses the application's configured scoring rules to produce an overall verdict.
- 📊 **Interactive dashboard:** Displays individual scores, the combined score, analysis notes, and a detailed report.
- 📄 **Report export:** Supports downloading analysis results when report export is enabled.

### Main Features

- Upload supported video, audio, or image files.
- Preview the uploaded media.
- Analyze available audio and visual signals separately.
- Display voice, face, and combined scores.
- Classify results as `REAL`, `FAKE`, or `UNCERTAIN`.
- Export analysis details for further review.
- Run locally or deploy to a compatible cloud platform.

> ⚠️ The result is a model-generated assessment, not definitive proof that media is real or fake.

---

## 🏗️ System Architecture

The application separates the uploaded media into the relevant analytical paths. Their outputs are then presented together through the dashboard.

### Architecture Flow Diagram

```text
                 ┌────────────────────────┐
                 │   Uploaded Media File  │
                 └────────────┬───────────┘
                              │
                 ┌────────────▼───────────┐
                 │ Media Processing Layer │
                 └────────────┬───────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
       ┌───────▼────────┐            ┌───────▼────────┐
       │  VIDEO STREAM  │            │  AUDIO STREAM │
       └───────┬────────┘            └───────┬────────┘
               │                             │
       ┌───────▼────────┐            ┌───────▼────────┐
       │ Frame Sampling │            │ Audio Track   │
       │                │            │ Extraction    │
       └───────┬────────┘            └───────┬────────┘
               │                             │
       ┌───────▼────────┐            ┌───────▼────────┐
       │ Face Tracker   │            │ Spectrogram / │
       │ Haar Cascade   │            │ Feature Parsing│
       └───────┬────────┘            └───────┬────────┘
               │                             │
       ┌───────▼────────┐            ┌───────▼────────┐
       │ Face Cropping  │            │ Speech Model  │
       │ and Tensor     │            │ Evaluation    │
       │ Conversion     │            │               │
       └───────┬────────┘            └───────┬────────┘
               │                             │
       ┌───────▼────────┐            ┌───────▼────────┐
       │ GendModel      │            │ Audio Score   │
       │ Inference      │            │               │
       └───────┬────────┘            └───────┬────────┘
               │                             │
       ┌───────▼────────┐                    │
       │ Visual Score   │                    │
       └───────┬────────┘                    │
               └──────────────┬──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Score Combination │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Final Prediction │
                    │ REAL / FAKE /     │
                    │ UNCERTAIN         │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Streamlit Dashboard│
                    │ and Report Export  │
                    └───────────────────┘
```

### 🎥 Video Detection Pipeline

1. The video is decoded into frames.
2. Selected frames are processed by the face-detection stage.
3. A Haar Cascade XML classifier can identify candidate face regions.
4. Detected faces are cropped, resized, and preprocessed.
5. The resulting image tensors are passed to the configured visual model.
6. The model output is converted into a visual score by the inference logic.

### 🧠 Understanding Image Tensors

Image models commonly process data in the following shape:

`[Batch_Size, Channels, Height, Width]`

For example, four RGB face images resized to 224 × 224 pixels could be represented as:

`[4, 3, 224, 224]`

- **Batch size:** Number of images processed together.
- **Channels:** Image channels, commonly RGB.
- **Height:** Image height in pixels.
- **Width:** Image width in pixels.

The preprocessing pipeline converts image pixels into numerical tensors that the neural network can process. Neural-network layers then apply learned weights and operations to extract features and generate classification outputs.

*The example dimensions are illustrative. The actual input shape must match the loaded model's configuration.*

### 🎙️ Audio Detection Pipeline

1. The application extracts or accesses the available audio track.
2. The audio is decoded and prepared for the model.
3. The configured preprocessing pipeline generates the expected audio representation.
4. The speech-authentication model evaluates acoustic characteristics.
5. The model output is converted into an audio score.
6. The score is displayed alongside the visual result.

### ⚖️ Final Verdict

The dashboard displays the combined score and applies the thresholds configured in the project.

- **REAL:** The combined score falls within the configured real range.
- **FAKE:** The combined score reaches the configured fake threshold.
- **UNCERTAIN:** The score falls between the configured thresholds, or the implementation cannot confidently resolve the result.

The exact score-combination formula depends on the implementation in the repository.

---

## 🧬 Deep Learning Models

### 1. Visual Detection Track

| Specification | Details |
|---|---|
| Target Model ID | `HoopitAI/video-deepfake-detection-GenD_CLIP_L_14_FF` |
| Architecture Class | `GendModel` |
| Visual Backbone | `openai/clip-vit-large-patch14` |
| Hidden Embedding Dimension | `hidden_size: 1024` |
| Verification Vertex Map | `vertex_num: 15` |
| Classification Outputs | `num_labels: 2` |
| Weights Format | `model.safetensors` |
| Target Training Benchmark | FaceForensics++ (FF++) |

The visual detection track is intended to evaluate facial information extracted from video frames. The configured model processes preprocessed face images and produces outputs used by the application's visual scoring logic.

The CLIP vision backbone is based on a Vision Transformer architecture that extracts feature representations from image patches. A task-specific model and classification head can use these features to distinguish between the configured classes.

**Dataset context:** FaceForensics++ is a benchmark for facial manipulation detection that includes manipulated videos generated using face-manipulation techniques. The exact checkpoint training history and dataset provenance should be verified against the model's documentation.

### 2. Audio Authentication Track

| Specification | Details |
|---|---|
| Target Configuration | `DF_Arena_500M_V_1` |
| Analytical Focus | Synthetic speech and voice authentication |
| Core Mechanism | Acoustic and spectral artifact analysis |
| Target Training Data | Multi-speaker voice-cloning and synthetic-speech datasets |
| Dashboard Output | Audio score |

The audio track is intended to identify acoustic patterns associated with genuine or generated speech. Depending on its implementation, the model may use spectral representations, acoustic frames, and learned speech features.

Synthetic speech can exhibit artifacts related to vocoders, spectral patterns, timing, or speech generation. However, these signals vary across generators, recording conditions, languages, and audio quality.

> **Note:** The model identifiers, parameters, and dataset descriptions above represent the intended configuration. Verify them against the actual checkpoints and model documentation before presenting them as independently confirmed facts.

### 🔐 Safe Model Weights

The `safetensors` format stores tensors without relying on Python pickle-based object deserialization. This helps reduce certain risks associated with loading serialized model objects.

However, a file format alone does not guarantee that a model is trustworthy, accurate, or correctly trained. Model weights should be downloaded from trusted sources.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web dashboard |
| PyTorch | Deep-learning inference |
| Transformers | Model and processor utilities |
| Safetensors | Model-weight loading |
| OpenCV | Video frame and face processing |
| NumPy | Numerical array processing |
| Pillow | Image processing |
| Scikit-learn | Statistical and evaluation utilities |
| FFmpeg | Media decoding support, when required |

---

## 📦 Dependencies & Version Control Matrix

The following versions form the requested dependency baseline. Confirm compatibility with the actual model repositories and deployment environment.

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | `==1.35.0` | Dashboard interface |
| `transformers` | `>=4.40.0` | Model loading utilities |
| `safetensors` | `>=0.4.3` | Safe tensor-weight loading |
| `torch` | `>=2.2.0` | Deep-learning runtime |
| `torchvision` | Compatible version | Vision utilities |
| `torchaudio` | Compatible version | Audio utilities |
| `opencv-python-headless` | `>=4.9.0` | Headless frame processing |
| `numpy` | `<2.0.0` | Array processing and compatibility |
| `pillow` | `>=10.2.0` | Image processing |
| `scikit-learn` | `>=1.4.0` | Statistical utilities |

### Why pin NumPy below 2.0?

Some packages and cloud environments have compatibility issues with NumPy 2.x. The `numpy<2.0.0` constraint can help maintain compatibility with dependencies that have not been validated against NumPy 2.

For reproducible installations, use compatible versions of PyTorch, torchvision, and torchaudio rather than installing arbitrary versions independently.

---

## 💻 Installation and Setup

### Prerequisites

- Python installed on your system.
- Git, if cloning the repository.
- Internet access if model weights must be downloaded.
- Enough memory and disk space for model loading and media processing.
- FFmpeg or the media-processing dependency configured by the project.

### Step 1: Open the project directory

If the repository is hosted on GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

If you already have the project, open the folder containing `app.py`, `detect.py`, and `settings.py` in VS Code.

### Step 2: Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install the dependencies

If your repository contains `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure the requirements file contains the compatible dependencies needed by your actual model implementation.

### Step 4: Run command-line analysis

The intended example command is:

```bash
python detect.py --dataset "Fake/sample_fake.mp4"
```

**Important:** This command works only if `detect.py` implements a command-line argument named `--dataset`. If your current `detect.py` only exposes an inference function such as `detect.analyze(path)`, use the dashboard workflow or implement a CLI wrapper separately.

Make sure the input file exists at the specified path.

### Step 5: Launch the Streamlit dashboard

Run the following command from the directory containing `app.py`:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Upload a supported media file and click **Analyze** to begin processing.

---

## ☁️ Cloud Deployment

The application can be deployed to a compatible platform such as Streamlit Community Cloud or Hugging Face Spaces, subject to model licensing, hardware, memory, and runtime limitations.

### 🐧 Why use `packages.txt`?

Some Linux cloud containers do not include the native system libraries required by OpenCV and its dependencies.

A `packages.txt` file can specify system packages that a supported platform should install during deployment.

Example:

```text
libgl1-mesa-glx
libglib2.0-0
```

- `libgl1-mesa-glx` provides OpenGL/GLX-related libraries on Linux distributions where this package is available.
- `libglib2.0-0` provides GLib runtime libraries used by many native applications and dependencies.

Package names can vary across Linux distributions. Some newer distributions use `libgl1` instead of `libgl1-mesa-glx`; use the package available for your deployment image.

### 🖥️ What is headless processing?

The `opencv-python-headless` package is designed for environments that do not require OpenCV's graphical user-interface features.

With the correct native libraries, OpenCV can decode video and process frames inside a Linux container without a physical monitor or desktop session.

> Headless processing does not automatically provide GPU acceleration or unlimited CPU, memory, disk space, or execution time.

### Cloud Deployment Checklist

- [ ] Add `requirements.txt` to the repository.
- [ ] Add `packages.txt` if the selected platform supports it.
- [ ] Configure the correct Streamlit entry point.
- [ ] Configure model access and secrets securely.
- [ ] Confirm model downloads fit the available storage and startup-time limits.
- [ ] Test audio and video inference in the deployed environment.
- [ ] Review upload limits, temporary file cleanup, and privacy requirements.

### Performance Recommendations

- Cache model initialization to avoid unnecessary reloads.
- Limit the number of sampled frames where appropriate.
- Avoid processing unnecessarily long media files.
- Handle model download and inference errors gracefully.
- Use a deployment instance with sufficient memory.
- Keep API tokens and credentials out of source control.
- Test cold-start time and CPU performance before publishing.

---

## 📊 Report and Results

The dashboard can present a report for the analyzed file, including available information such as:

- Uploaded file name and format.
- Analysis timestamp and duration.
- Final verdict and combined score.
- Audio score and visual score.
- Speech duration and analysis windows, if returned by the audio model.
- Sampled frame counts and face-detection details, if returned by the visual model.
- Model notes and configured thresholds.
- Downloadable report data, if export is enabled.

### Understanding the Scores

| Result | Interpretation |
|---|---|
| Combined Score | Output of the configured score-combination logic |
| Voice Score | Audio model output as interpreted by the application |
| Face Score | Visual model output as interpreted by the application |
| REAL | Meets the configured real-verdict rule |
| FAKE | Meets the configured fake-verdict rule |
| UNCERTAIN | Falls within the configured uncertainty region or cannot be resolved |

Scores should not automatically be interpreted as calibrated probabilities. Their meaning depends on model outputs, post-processing, and evaluation.

---

## ⚠️ Limitations

- New or unseen deepfake-generation methods may not be detected reliably.
- Video compression and low-resolution frames can reduce visual accuracy.
- Background noise, music, and short speech segments can affect audio analysis.
- A face detector only identifies candidate face regions; it does not determine authenticity.
- The audio and visual models may disagree.
- Performance can vary across languages, speakers, recording devices, and media sources.
- A model's score does not constitute forensic proof.

Use this application as an investigative aid rather than the sole basis for consequential decisions.

---

## 🔒 Privacy and Responsible Use

- Obtain appropriate permission before analyzing other people's media.
- Treat uploaded audio and video as potentially sensitive data.
- Do not commit private media, credentials, or access tokens to GitHub.
- Review temporary-file cleanup and data-retention behavior before deployment.
- Check the licenses for third-party models and datasets.
- Evaluate false-positive and false-negative rates before making reliability claims.

---

## 🗺️ Future Improvements

- [ ] Add a documented command-line interface for batch inference.
- [ ] Evaluate both models on representative test datasets.
- [ ] Publish precision, recall, F1-score, and other relevant metrics.
- [ ] Calibrate model scores and document threshold selection.
- [ ] Add automated tests for preprocessing and score combination.
- [ ] Add JSON report export alongside CSV.
- [ ] Improve cloud inference speed and memory efficiency.
- [ ] Document checkpoint provenance and licenses.

---

## 🤝 Contributing

Contributions and improvements are welcome.

1. Create a feature branch.
2. Keep inference changes separate from UI changes where possible.
3. Test changes with a small, non-sensitive media sample.
4. Document new dependencies and configuration settings.
5. Do not commit private media, credentials, or large model weights without an explicit repository policy.

---

## 📜 License

Add your project's license here. Also review the separate licenses and usage conditions for any third-party model weights, datasets, or pretrained backbones used by this project.

---

<div align="center">

**🛡️ Multi-Modal Deepfake Detection Dashboard**

*Analyze the signals. Understand the uncertainty. Verify the evidence.*

</div>