from pathlib import Path
import subprocess

INPUT_ROOT = Path("dataset/final")
OUTPUT_ROOT = Path("processed/audio")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for category in INPUT_ROOT.iterdir():

    if not category.is_dir():
        continue

    out_dir = OUTPUT_ROOT / category.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for video in category.glob("*.mp4"):

        wav_path = out_dir / f"{video.stem}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video),
            "-ac", "1",
            "-ar", "16000",
            str(wav_path)
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"Extracted: {wav_path.name}")