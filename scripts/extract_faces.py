import os
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================

MANIFEST = "metadata/debug_manifest.csv"
FRAMES_DIR = "frames"
FACES_DIR = "faces"
FAILED_DIR = "failed_faces"
COMPRESSION_LEVELS = ["original", "crf23", "crf28", "crf35"]

# ==========================

mtcnn = MTCNN(image_size=224, margin=20, keep_all=False)

df = pd.read_csv(MANIFEST)

total_saved = 0
total_failed = 0

for _, row in df.iterrows():
    filename = os.path.basename(row["video_path"])
    video_name = os.path.splitext(filename)[0]
    label = "real" if row["label"] == 1 else "fake"
    split = row["split"]

    for level in COMPRESSION_LEVELS:
        frames_folder = os.path.join(FRAMES_DIR, level)
        save_folder = os.path.join(FACES_DIR, split, label, level)
        failed_folder = os.path.join(FAILED_DIR, level)

        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(failed_folder, exist_ok=True)

        frame_files = [
            f for f in os.listdir(frames_folder)
            if f.startswith(video_name) and f.endswith(".jpg")
        ]

        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            img = Image.open(frame_path).convert("RGB")

            face = mtcnn(img)

            if face is not None:
                face_img = Image.fromarray(
                    (face.permute(1, 2, 0).numpy() * 128 + 127.5)
                    .clip(0, 255)
                    .astype("uint8")
                )
                face_img.save(os.path.join(save_folder, frame_file))
                total_saved += 1
            else:
                img.save(os.path.join(failed_folder, frame_file))
                total_failed += 1

        print(f"{level} | {video_name}: done")

print(f"\nFinished! Saved: {total_saved} | Failed: {total_failed}")