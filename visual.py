"""
Face / frame deepfake detector (pretrained, no training).

Pipeline: sample frames -> find the largest face -> crop -> score each crop with a
pretrained face-forgery model -> average. Returns None if no face is found.
"""
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModel 
from settings import IMAGE_EXT, VIDEO_MODEL

# ------------------------------------------------------------------ SETTINGS
MAX_FRAMES = 24        # frames analysed per video (spread evenly over the video)
FACE_MARGIN = 0.30     # extra area around the detected face box
MIN_FACE = 60          # ignore faces smaller than this many pixels
BATCH = 8
# ---------------------------------------------------------------------------

_model = None
#AutoModel.from_pretrained(VIDEO_MODEL, trust_remote_code=True)
_device = "cpu"
_cascade = None
import os
       
def load_model():
    global _model, _device, _processor
    if _model is None:
        import torch
        from transformers import AutoImageProcessor, AutoConfig
        # 1. This is the missing import that clears your yellow underline warning!
        from transformers.utils import cached_file
        from safetensors.torch import load_file
        
        # Manually pull the custom model class from the clean file in your workspace
        from modelling_gend import GendModel
        
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading custom pre-trained deepfake model: {VIDEO_MODEL} ...")
        
        # Extract the configuration metadata parameters
        config = AutoConfig.from_pretrained(VIDEO_MODEL, trust_remote_code=True)
        
        # Download just the weight tensors cleanly without using broken factory loaders
        weights_path = cached_file(VIDEO_MODEL, "model.safetensors")
        state_dict = load_file(weights_path, device="cpu")
        
        # Initialize the architecture manually and bind the weights cleanly
        _model = GendModel(config)
        _model.load_state_dict(state_dict, strict=False)
        
        _processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _model.eval().to(_device)
    return _model

def _detector():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return _cascade


def sample_frames(path):
    """Return up to MAX_FRAMES BGR frames spread evenly across the video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(fps)), 1)                 # at least ~1 frame per second
    if total > 0:
        step = max(step, total // MAX_FRAMES)
    frames, i = [], 0
    while len(frames) < MAX_FRAMES:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        i += 1
    cap.release()
    return frames


def crop_face(frame_bgr):
    """Largest face in the frame as an RGB PIL image, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(MIN_FACE, MIN_FACE))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(FACE_MARGIN * max(w, h))
    H, W = frame_bgr.shape[:2]
    crop = frame_bgr[max(y - m, 0):min(y + h + m, H), max(x - m, 0):min(x + w + m, W)]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def score_crops(crops):
    import torch
    from transformers import AutoImageProcessor
    model = load_model()
    global _device
    # Initialize the processor directly inside the scoring function using your config variable
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")# Update any other hidden location to match this exact string line:

    probs = []
    
    for i in range(0, len(crops), BATCH):
        # 1. This is the missing line that creates 'batch'! Make sure it is here:
        batch = torch.stack([processor(c, return_tensors="pt")["pixel_values"].squeeze(0) for c in crops[i:i + BATCH]])
        
        with torch.no_grad():
           with torch.no_grad():
            logits = model(batch.to(_device))
            # Handle standard Hugging Face model outputs if wrapped in an object attribute
            if hasattr(logits, "logits"):
                logits = logits.logits
                
        # Calculate final software threshold predictions safely
        probs += logits.softmax(dim=-1)[:, 1].cpu().tolist()
    return probs

def is_cartoon(frames):
    """
    Checks the first 3 frames of the video. 
    Returns True if it detects a cartoon, anime, or 2D/3D animation.
    """
    try:
        from transformers import pipeline
        # Initializes a lightweight, fast animation classification eye
        classifier = pipeline("image-classification", model="skytnt/anime-detector")
        
        cartoon_votes = 0
        # Check a quick sample of up to 3 frames to avoid slowing things down
        test_frames = frames[:3] 
        
        for frame in test_frames:
            # Convert OpenCV BGR arrays to a clean PIL RGB image format for the pipeline
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predictions = classifier(pil_img)
            
            # The model outputs standard labels like 'anime', 'cartoon', or 'real'
            top_prediction = predictions[0]['label']
            if top_prediction in ['anime', 'cartoon']:
                cartoon_votes += 1
                
        # If the majority of checked frames look like animation, flag it
        return cartoon_votes >= 2
    except Exception as e:
        # If the check ever fails (e.g., network timeout), default safely to letting the pipeline continue
        print(f"\n[System Notice] Cartoon pre-check skipped: {e}")
        return False
    
def analyze(path):
    """Video or image path -> {'score', 'frames_used', 'frames_checked'} or None if no face."""
    if path.lower().endswith(IMAGE_EXT):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("Could not read the image.")
        frames = [img]
    else:
        frames = sample_frames(path)
    crops = [c for c in (crop_face(f) for f in frames) if c is not None]
    if not crops:
        return None
    probs = score_crops(crops)
    return {"score": float(np.mean(probs)), "max_frame_score": float(np.max(probs)),
            "frames_used": len(crops), "frames_checked": len(frames)}

