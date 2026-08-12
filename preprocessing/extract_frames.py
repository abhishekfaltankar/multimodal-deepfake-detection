# preprocessing/extract_frames.py
import cv2
import os
import argparse

def extract_frames(video_dir, output_dir, frame_interval=10):
    os.makedirs(output_dir, exist_ok=True)

    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(videos)} videos in {video_dir}")

    for vid_name in videos:
        vid_path = os.path.join(video_dir, vid_name)
        vid_basename = os.path.splitext(vid_name)[0]
        save_folder = os.path.join(output_dir, vid_basename)
        os.makedirs(save_folder, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  [SKIP] Could not open {vid_name}")
            continue

        frame_idx = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                out_path = os.path.join(save_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            frame_idx += 1

        cap.release()
        print(f"  {vid_name}: {saved_count} frames saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", required=True, help="e.g. faceswap or edited")
    parser.add_argument("--frame_interval", type=int, default=10)
    args = parser.parse_args()

    video_dir = f"dataset/final/{args.class_name}"
    output_dir = f"processed/frames/{args.class_name}"

    extract_frames(video_dir, output_dir, args.frame_interval)
    print(f"\nDONE: frames extracted for '{args.class_name}'")