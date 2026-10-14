"""
Light settings file (no heavy imports) so the web page opens instantly.
Change models, thresholds and the voice-only switch here.
"""
import shutil

AUDIO_MODEL = "Speech-Arena-2025/DF_Arena_500M_V_1"                  # bigger/better: ..._1B_V_1 (needs a GPU)
VIDEO_MODEL = "HoopitAI/video-deepfake-detection-GenD_CLIP_L_14_FF"  # face model (class 0 real, 1 fake)
#"HoopitAI/video-deepfake-detection-GenD_DINOv3_L_FF"
ENABLE_FACE = True    # False = voice-only mode: faster, and skips the face model download

FAKE_ABOVE = 0.60       # combined fake score >= this -> FAKE
REAL_BELOW = 0.40       # combined fake score <= this -> REAL (in between -> UNCERTAIN)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ffmpeg_exe():
    """ffmpeg from the system PATH, else the copy bundled with `pip install imageio-ffmpeg`."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None