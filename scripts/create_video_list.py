import os
import csv

BASE_DIR = r"C:\Users\niran\OneDrive\Desktop\MINI PROJECT\Celeb_dataset" # 👈 CHANGE THIS

data = []

def add_videos(folder, label, source):
    folder_path = os.path.join(BASE_DIR, folder)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".mp4"):
                rel_path = os.path.join(folder, f).replace("\\", "/")
                data.append([rel_path, label, source])

# REAL VIDEOS
add_videos("Celeb-real", 1, "celeb-real")
add_videos("YouTube-real", 1, "youtube-real")

# FAKE VIDEOS
add_videos("Celeb-synthesis", 0, "celeb-synthesis")

# SAVE CSV
output_file = "metadata/video_list.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "label", "source"])
    writer.writerows(data)

print(f"Done! Saved {len(data)} videos to {output_file}")
