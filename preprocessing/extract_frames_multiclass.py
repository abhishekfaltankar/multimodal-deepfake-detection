from pathlib import Path
import cv2

ROOT = Path("dataset/final")
OUT = Path("processed/frames")

CLASSES = [
    "real",
    "ai_video",
    "faceswap",
    "edited",
    "voiceclone",
    "audio_video_fake"
]

FRAME_STEP = 10  # save every 10th frame


def extract(video_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % FRAME_STEP == 0:
            name = f"{video_path.stem}_{saved:04d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            saved += 1

        idx += 1

    cap.release()
    print(f"{video_path.name} -> {saved} frames")


for cls in CLASSES:
    src = ROOT / cls
    dst = OUT / cls

    dst.mkdir(parents=True, exist_ok=True)

    videos = list(src.glob("*.mp4"))
    print(f"\\n[{cls}] {len(videos)} videos found")

    for video in videos:
        extract(video, dst)

print("\\nFrame extraction completed.")