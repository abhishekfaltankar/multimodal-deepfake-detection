# preprocessing/sync_multiclass.py
import shutil
from pathlib import Path

classes = ["real", "ai_video", "faceswap", "edited", "voiceclone", "audio_video_fake"]

faces_root = Path("processed/faces")
multiclass_root = Path("processed/multiclass")

for cls in classes:
    src = faces_root / cls
    dst = multiclass_root / cls

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in src.glob("*.jpg"):
        shutil.copy2(img_path, dst / img_path.name)
        count += 1

    print(f"{cls}: {count} images copied to multiclass")

print("\nDONE syncing multiclass folder")