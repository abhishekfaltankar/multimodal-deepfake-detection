import cv2
from pathlib import Path

print(cv2.__version__)
print(cv2.__file__)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(frames_root, faces_root):
    frames_root = Path(frames_root)
    faces_root = Path(faces_root)
    faces_root.mkdir(parents=True, exist_ok=True)

    total_faces = 0

    for video_folder in frames_root.iterdir():
        if not video_folder.is_dir():
            continue

        output_folder = faces_root / video_folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in video_folder.glob("*.jpg"):
            image = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for i, (x, y, w, h) in enumerate(faces):
                face = image[y:y+h, x:x+w]

                face_file = output_folder / f"{frame_path.stem}_face{i}.jpg"
                cv2.imwrite(str(face_file), face)
                total_faces += 1

    return total_faces

# Extract REAL faces
real_faces = extract_faces(
    "dataset/frames_real",
    "dataset/faces_real"
)

# Extract FAKE faces
fake_faces = extract_faces(
    "dataset/frames_fake",
    "dataset/faces_fake"
)

print(f"Real faces extracted: {real_faces}")
print(f"Fake faces extracted: {fake_faces}")
print("FF++ face extraction completed.")