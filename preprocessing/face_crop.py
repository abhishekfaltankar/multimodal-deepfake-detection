# preprocessing/face_crop.py
from pathlib import Path
from ultralytics import YOLO
import cv2
import shutil

model = YOLO(r"weights/yolov8n-face.pt")

def crop_faces(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    # wipe stale output from previous runs/naming schemes
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0

    for i, img_path in enumerate(input_root.rglob("*.jpg")):
        if i % 3 != 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)

        if len(results[0].boxes) > 0:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
            face = img[y1:y2, x1:x2]

            if face.size != 0:
                out_name = f"{img_path.parent.name}_{img_path.name}"
                cv2.imwrite(str(output_root / out_name), face)
                count += 1

    print(f"{input_root.name}: {count} faces")

if __name__ == "__main__":
    crop_faces("processed/frames/real", "processed/faces/real")
    crop_faces("processed/frames/ai_video", "processed/faces/ai_video")
    crop_faces("processed/frames/faceswap", "processed/faces/faceswap")
    crop_faces("processed/frames/edited", "processed/faces/edited")
    crop_faces("processed/frames/voiceclone", "processed/faces/voiceclone")
    crop_faces("processed/frames/audio_video_fake", "processed/faces/audio_video_fake")