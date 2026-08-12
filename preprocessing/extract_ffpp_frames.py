import cv2
from pathlib import Path

def extract_frames(video_dir, output_root):
    video_dir = Path(video_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for video_path in video_dir.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0

        video_output = output_root / video_path.stem
        video_output.mkdir(parents=True, exist_ok=True)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Save every 10th frame
            if frame_count % 10 == 0:
                frame_file = video_output / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_file), frame)

            frame_count += 1

        cap.release()
        print(f"Processed {video_path.name} ({frame_count} frames)")

# REAL videos
extract_frames(
    "dataset/ffpp_sample/original_sequences/youtube/c40/videos",
    "dataset/frames_real"
)

# FAKE videos
extract_frames(
    "dataset/ffpp_sample/manipulated_sequences/Deepfakes/c40/videos",
    "dataset/frames_fake"
)

print("FaceForensics++ frame extraction completed.")