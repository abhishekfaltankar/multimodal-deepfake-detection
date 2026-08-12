from pathlib import Path

real_dir = Path("dataset/prepared/real")
fake_dir = Path("dataset/prepared/fake")

real_count = len(list(real_dir.glob("*.jpg")))
fake_count = len(list(fake_dir.glob("*.jpg")))

print("DATASET SUMMARY")
print("-" * 30)
print(f"Real images : {real_count}")
print(f"Fake images : {fake_count}")
print(f"Total images: {real_count + fake_count}")