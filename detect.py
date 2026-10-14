"""
Deepfake checker for VIDEO files: checks the VOICE and the FACE, then combines both.
Result: REAL / FAKE / UNCERTAIN (pretrained models, no training).
"""

import os
import subprocess
import sys
import tempfile
import wave
import shutil

import numpy as np
import torch
from transformers import pipeline

import visual
from settings import AUDIO_MODEL, ENABLE_FACE, FAKE_ABOVE, REAL_BELOW, ffmpeg_exe

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
SR = 16000
WINDOW_SEC = 4
HOP_SEC = 4
MAX_WINDOWS = 15

FAKE_IF_ANY_ABOVE = 0.90
WEIGHTS = {"audio": 0.5, "video": 0.5}

CLIP, HOP = WINDOW_SEC * SR, HOP_SEC * SR

_audio_pipe = None
_audio_error = None
_vad = None


# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
def load_models():
    """Load the custom audio anti-spoofing pipeline."""
    global _audio_pipe, _audio_error

    if _audio_pipe is None and _audio_error is None:
        try:
            print(f"Loading voice model: {AUDIO_MODEL} ...")

            device = 0 if torch.cuda.is_available() else -1

            _audio_pipe = pipeline(
                "antispoofing",
                model=AUDIO_MODEL,
                trust_remote_code=True,
                device=device,
            )

            print("Voice model loaded successfully.")

        except Exception as exc:
            _audio_error = exc
            print(f"Voice model could not be loaded: {exc}")

    # Audio loading failure should not prevent face analysis.
    if ENABLE_FACE:
        visual.load_model()


# ------------------------------------------------------------------
# AUDIO EXTRACTION
# ------------------------------------------------------------------
def to_wav(path):
    """Extract 16 kHz mono audio from a video or audio file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found. Install it or run: "
            "python -m pip install imageio-ffmpeg"
        )

    temp_dir = tempfile.mkdtemp()
    out = os.path.join(temp_dir, "audio.wav")

    try:
        res = subprocess.run(
            [
                exe, "-y", "-i", path,
                "-vn", "-ac", "1", "-ar", str(SR),
                "-sample_fmt", "s16", out,
            ],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Could not run ffmpeg: {exc}") from exc

    if res.returncode != 0 or not os.path.exists(out):
        err = (res.stderr or "").strip()
        shutil.rmtree(temp_dir, ignore_errors=True)

        if (
            "does not contain any stream" in err
            or "Output file #0 does not contain" in err
        ):
            raise ValueError("This file does not contain an audio track.")

        raise RuntimeError(
            "ffmpeg could not extract audio: "
            + " | ".join(err.splitlines()[-3:])[:300]
        )

    return out


# ------------------------------------------------------------------
# AUDIO PREPROCESSING
# ------------------------------------------------------------------
def _energy_vad(y):
    """Simple energy-based silence reduction."""
    frame_size = 480
    count = len(y) // frame_size

    if count == 0:
        return y

    frames = y[:count * frame_size].reshape(count, frame_size)
    rms = np.sqrt((frames ** 2).mean(axis=1))

    if rms.max() == 0:
        return y

    keep = rms > rms.max() * (10 ** (-30 / 20))
    return frames[keep].reshape(-1) if keep.any() else y


def speech_only(y):
    """Use Silero VAD if available; otherwise use energy-based filtering."""
    global _vad

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        if _vad is None:
            _vad = load_silero_vad()

        timestamps = get_speech_timestamps(
            torch.from_numpy(y),
            _vad,
            sampling_rate=SR,
        )

        if timestamps:
            kept = np.concatenate(
                [y[t["start"]:t["end"]] for t in timestamps]
            )
            if len(kept) >= SR:
                return kept

    except Exception:
        pass

    return _energy_vad(y)


def load_wav(path):
    """Read a mono 16-bit PCM WAV file."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError("Unexpected audio format; expected 16-bit PCM.")

    if rate != SR:
        raise ValueError(f"Expected {SR} Hz audio, got {rate} Hz.")

    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)

    return y, rate


def make_windows(y):
    """Split audio into fixed-length windows."""
    if len(y) < CLIP:
        y = np.pad(y, (0, CLIP - len(y)), mode="constant")

    starts = list(range(0, len(y) - CLIP + 1, HOP))

    if len(starts) > MAX_WINDOWS:
        indices = np.linspace(
            0, len(starts) - 1, MAX_WINDOWS
        ).astype(int)
        starts = [starts[i] for i in indices]

    return [y[start:start + CLIP] for start in starts]


# ------------------------------------------------------------------
# AUDIO SCORE INTERPRETATION
# ------------------------------------------------------------------
def spoof_prob(output):
    """Extract the spoof probability from a custom pipeline result."""
    if isinstance(output, list):
        if not output:
            raise ValueError("Audio model returned an empty list.")
        output = output[0]

    if not isinstance(output, dict):
        raise ValueError(
            f"Unexpected audio model output type: {type(output).__name__}"
        )

    all_scores = output.get("all_scores", {})

    if isinstance(all_scores, dict):
        scores = {
            str(key).lower(): float(value)
            for key, value in all_scores.items()
        }

        for key in ("spoof", "fake"):
            if key in scores:
                return scores[key]

        for key in ("bonafide", "bona fide", "real"):
            if key in scores:
                return 1.0 - scores[key]

    label = str(output.get("label", "")).lower()

    if label in ("spoof", "fake"):
        return 1.0

    if label in ("bonafide", "bona fide", "real"):
        return 0.0

    raise ValueError(
        f"Could not identify spoof score in model output: {output}"
    )


# ------------------------------------------------------------------
# AUDIO ANALYSIS
# ------------------------------------------------------------------
def analyze_audio(path):
    """Analyze audio using the custom antispoofing pipeline."""
    load_models()

    if _audio_pipe is None:
        raise RuntimeError(f"Audio model failed to load: {_audio_error}")

    wav_path = to_wav(path)

    try:
        y, _ = load_wav(wav_path)

        if len(y) < SR // 2:
            raise ValueError("No usable audio was found.")

        y = speech_only(y.astype(np.float32))
        windows = make_windows(y)
        probabilities = []

        for window in windows:
            # The custom pipeline may accept a raw NumPy waveform.
            # Runtime output must be checked against this model's API.
            result = _audio_pipe(
                window.astype(np.float32),
                sampling_rate=SR,
            )

            probabilities.append(spoof_prob(result))

        return {
            "score": float(np.mean(probabilities)),
            "speech_sec": len(y) / SR,
            "windows": len(windows),
        }

    finally:
        shutil.rmtree(
            os.path.dirname(wav_path),
            ignore_errors=True,
        )


# ------------------------------------------------------------------
# SCORE FUSION
# ------------------------------------------------------------------
def fuse(results, notes=None):
    if not results:
        raise RuntimeError("No modality produced a usable result.")

    weights = {key: WEIGHTS[key] for key in results}
    total_weight = sum(weights.values())

    score = sum(
        results[key]["score"] * weights[key]
        for key in results
    ) / total_weight

    if (
        score >= FAKE_ABOVE
        or any(
            result["score"] >= FAKE_IF_ANY_ABOVE
            for result in results.values()
        )
    ):
        verdict = "FAKE"
    elif score <= REAL_BELOW:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    return {
        "verdict": verdict,
        "score": float(score),
        "audio": results.get("audio"),
        "video": results.get("video"),
        "notes": notes or [],
    }


# ------------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------------
def analyze(path):
    """Analyze a video, audio file, or image."""
    results = {}
    notes = []

    is_image = path.lower().endswith(visual.IMAGE_EXT)

    if not is_image:
        try:
            results["audio"] = analyze_audio(path)
        except Exception as exc:
            notes.append(f"Voice not checked: {exc}")

    if ENABLE_FACE:
        try:
            visual.load_model()
            video_result = visual.analyze(path)

            if video_result:
                results["video"] = video_result
            else:
                notes.append(
                    "No face found, so the face check was skipped."
                )

        except Exception as exc:
            notes.append(f"Face check skipped: {exc}")

    elif is_image:
        notes.append(
            "Face checking is disabled in settings.py."
        )

    if not results:
        raise RuntimeError(
            "Could not analyse this file. " + " ".join(notes)
        )

    return fuse(results, notes)


# ------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ------------------------------------------------------------------
def check(path):
    result = analyze(path)

    print("\n" + "-" * 48)

    if result["audio"]:
        audio = result["audio"]
        print(
            f"  Voice : fake score {audio['score']:.2%} "
            f"({audio['speech_sec']:.1f}s of audio)"
        )

    if result["video"]:
        video = result["video"]
        print(
            f"  Face  : fake score {video['score']:.2%} "
            f"({video['frames_used']} frames with a face)"
        )

    for note in result["notes"]:
        print(f"  Note  : {note}")

    print(
        f"  RESULT: {result['verdict']} "
        f"(combined fake score {result['score']:.2%})"
    )
    print("-" * 48)


def main():
    if len(sys.argv) > 1:
        check(sys.argv[1].strip('"'))
        return

    while True:
        path = input(
            "\nEnter video file path (q to quit): "
        ).strip().strip('"')

        if path.lower() == "q":
            break

        try:
            check(path)
        except Exception as exc:
            print("Error:", exc)


if __name__ == "__main__":
    main()
