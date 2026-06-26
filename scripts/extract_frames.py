import os
import cv2
import pandas as pd

# ==========================
# CONFIGURATION
# ==========================

MANIFEST = "metadata/debug_manifest.csv"
COMPRESSED_DIR = "compressed_videos"
FRAMES_DIR = "frames"
FRAME_INTERVAL = 10

COMPRESSION_LEVELS = ["original", "crf23", "crf28", "crf35"]

# ==========================

df = pd.read_csv(MANIFEST)

for _, row in df.iterrows():
    filename = os.path.basename(row["video_path"])
    video_name = os.path.splitext(filename)[0]

    for level in COMPRESSION_LEVELS:
        video_path = os.path.join(COMPRESSED_DIR, level, filename)

        save_folder = os.path.join(FRAMES_DIR, level)
        os.makedirs(save_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_INTERVAL == 0:
                frame_filename = f"{video_name}_frame{frame_count:05d}.jpg"
                cv2.imwrite(os.path.join(save_folder, frame_filename), frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"{level} | {video_name}: {saved_count} frames extracted")

print("\nFinished!")