import cv2
from pathlib import Path
from ultralytics import YOLO
import sys

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
OUTPUT_DIR = Path("debug_faces")
OUTPUT_DIR.mkdir(exist_ok=True)

face_model = YOLO("weights/yolov8n-face.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Could not open {VIDEO_PATH}")
    sys.exit(1)

# print basic video info
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video info: {width}x{height} @ {fps:.1f}fps, {frame_count} total frames")

frame_idx = 0
saved = 0

while saved < 10:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 10 == 0:
        results = face_model(frame, verbose=False)

        if len(results[0].boxes) > 0:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                out_path = OUTPUT_DIR / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(out_path), face)
                print(f"Saved {out_path} | face size: {face.shape[1]}x{face.shape[0]} | box conf: {float(results[0].boxes[0].conf[0]):.2f}")
                saved += 1
        else:
            print(f"Frame {frame_idx}: no face detected")

    frame_idx += 1

cap.release()
print(f"\nSaved {saved} face crops to {OUTPUT_DIR}/")