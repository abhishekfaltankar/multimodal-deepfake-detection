"""
Deepfake checker for VIDEO files: checks the VOICE and the FACE, then combines both.
Result: REAL / FAKE / UNCERTAIN (pretrained models, no training).

Install once:
    pip install -r requirements.txt        (+ ffmpeg installed on your computer)

Run:
    python detect.py                 -> asks for a video path again and again
    python detect.py clip.mp4        -> checks one video and exits
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import torch
from transformers import pipeline

import visual
from settings import AUDIO_MODEL, ENABLE_FACE, FAKE_ABOVE, REAL_BELOW, ffmpeg_exe

# ------------------------------------------------------------------ SETTINGS
# (models, thresholds and the voice-only switch are in settings.py)
SR = 16000            # sample rate the voice model expects
WINDOW_SEC = 4        # the voice model reads ~4 s at a time
HOP_SEC = 4           # step between windows
MAX_WINDOWS = 15      # cap for long videos (keeps CPU time reasonable)

FAKE_IF_ANY_ABOVE = 0.90   # one part (voice or face) very sure it's fake -> FAKE
WEIGHTS = {"audio": 0.5, "video": 0.5}   # tune with evaluate.py
# ---------------------------------------------------------------------------

CLIP, HOP = WINDOW_SEC * SR, HOP_SEC * SR
_audio_pipe, _vad = None, None


def load_models():
    global _audio_pipe
    if _audio_pipe is None:
        from transformers import AutoConfig, pipeline
        import torch
        
        print(f"Loading voice model: {AUDIO_MODEL} ...")
        
        # 1. Fetch the architecture settings blueprint securely first to bypass path errors
        config = AutoConfig.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
        
        # 2. Inject both the model ID and the config map blueprint into the pipeline object
        _audio_pipe = pipeline(
            "audio-classification", 
            model=AUDIO_MODEL, 
            config=config,
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1
        )
        
    if ENABLE_FACE:
        visual.load_model()


# ------------------------------------------------------------------ voice part
def to_wav(path):
    """Video/audio file -> 16 kHz mono wav (needs ffmpeg). Gives a clear reason if it fails."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}  (check the path; use the full path or one "
                                f"inside dataset\\real or dataset\\fake)")
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found. Run: python -m pip install imageio-ffmpeg")
    out = os.path.join(tempfile.mkdtemp(), "audio.wav")
    try:
        res = subprocess.run([exe, "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), out],
                             capture_output=True, text=True)
    except OSError as e:   # e.g. Windows blocked the ffmpeg program
        raise RuntimeError(f"Windows could not run ffmpeg ({e}). If it mentions an Application "
                           "Control policy, install the full ffmpeg: winget install Gyan.FFmpeg")
    if res.returncode != 0 or not os.path.exists(out):
        err = (res.stderr or "").strip()
        if "does not contain any stream" in err or "Output file #0 does not contain" in err:
            raise ValueError("this video has no audio track (it is silent)")
        raise RuntimeError("ffmpeg could not read the audio: " + " | ".join(err.splitlines()[-3:])[:300])
    return out


def speech_only(y):
    """Keep only the speaking parts (silero-vad if installed, else simple silence removal)."""
    global _vad
    kept = None
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        if _vad is None:
            _vad = load_silero_vad()
        ts = get_speech_timestamps(torch.from_numpy(y), _vad, sampling_rate=SR)
        if ts:
            kept = np.concatenate([y[t["start"]:t["end"]] for t in ts])
    except Exception:
        kept = _energy_vad(y)
    return y if kept is None or len(kept) < SR else kept


def _energy_vad(y):
    """Simple silence removal: keep 30 ms frames that are within 30 dB of the loudest frame."""
    f = 480
    n = len(y) // f
    if n == 0:
        return y
    frames = y[: n * f].reshape(n, f)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    keep = rms > rms.max() * 10 ** (-30 / 20)
    return frames[keep].reshape(-1) if keep.any() else y


def load_wav(path):
    """Read the 16 kHz mono 16-bit wav that ffmpeg made (pure standard library)."""
    with wave.open(path, "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("unexpected audio format")
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    return y, rate


def make_windows(y):
    if len(y) < CLIP:
        y = np.pad(y, (0, CLIP - len(y)), mode="wrap")
    starts = list(range(0, len(y) - CLIP + 1, HOP))
    if len(starts) > MAX_WINDOWS:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, MAX_WINDOWS).astype(int)]
    return [y[s:s + CLIP] for s in starts]


def spoof_prob(out):
    """Turn the voice model's output ({'label': ..., 'all_scores': {...}}) into P(fake)."""
    if isinstance(out, list):
        out = out[0]
    scores = {str(k).lower(): float(v) for k, v in (out.get("all_scores") or {}).items()}
    for k in ("spoof", "fake"):
        if k in scores:
            return scores[k]
    for k in ("bonafide", "real"):
        if k in scores:
            return 1.0 - scores[k]
    label = str(out.get("label", "")).lower()
    if label in ("spoof", "fake"):
        return 1.0
    if label in ("bonafide", "real"):
        return 0.0
    raise ValueError(f"Unexpected output from the voice model: {out}")


def analyze_audio(path):
    load_models()
    y, _ = load_wav(to_wav(path))
    if len(y) < SR // 2:
        raise ValueError("no usable audio")
    y = speech_only(y.astype(np.float32))
    windows = make_windows(y)
    probs = [spoof_prob(_audio_pipe(w)) for w in windows]
    return {"score": float(np.mean(probs)), "speech_sec": len(y) / SR, "windows": len(windows)}


# ------------------------------------------------------------------ combine
def fuse(results, notes=None):
    w = {k: WEIGHTS[k] for k in results}
    score = sum(results[k]["score"] * w[k] for k in results) / sum(w.values())
    if score >= FAKE_ABOVE or any(r["score"] >= FAKE_IF_ANY_ABOVE for r in results.values()):
        verdict = "FAKE"
    elif score <= REAL_BELOW:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"
    return {"verdict": verdict, "score": float(score),
            "audio": results.get("audio"), "video": results.get("video"),
            "notes": notes or []}


def analyze(path):
    """Video (or audio / image) path -> dict with verdict, combined score, voice part, face part."""
    load_models()
    results, notes = {}, []
    is_image = path.lower().endswith(visual.IMAGE_EXT)

    if not is_image:
        try:
            results["audio"] = analyze_audio(path)
        except Exception as e:
            notes.append(f"Voice not checked: {e}")
    if ENABLE_FACE:
        try:
            v = visual.analyze(path)
            if v:
                results["video"] = v
            else:
                notes.append("No face found, so the face check was skipped.")
        except Exception as e:
            notes.append(f"Face check skipped: {e}")
    elif is_image:
        notes.append("The face check is turned off in settings.py, and an image has no voice.")

    if not results:
        raise RuntimeError("Could not analyse this file. " + " ".join(notes))
    return fuse(results, notes)


# ------------------------------------------------------------------ command line
def check(path):
    r = analyze(path)
    print("\n" + "-" * 48)
    if r["audio"]:
        print(f"  Voice : fake score {r['audio']['score']:.2%}  ({r['audio']['speech_sec']:.1f}s of speech)")
    if r["video"]:
        print(f"  Face  : fake score {r['video']['score']:.2%}  ({r['video']['frames_used']} frames with a face)")
    for n in r["notes"]:
        print(f"  Note  : {n}")
    print(f"  RESULT: {r['verdict']}  (combined fake score {r['score']:.2%})")
    print("-" * 48)


def main():
    load_models()
    if len(sys.argv) > 1:
        check(sys.argv[1].strip('"'))
        return
    while True:
        path = input("\nEnter video file path (q to quit): ").strip().strip('"')
        if path.lower() == "q":
            break
        try:
            check(path)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
