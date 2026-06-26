import os
import subprocess
import shutil
import pandas as pd

DATASET_DIR = r"C:\Users\niran\OneDrive\Desktop\MINI PROJECT\Celeb_dataset"
MANIFEST = "metadata/debug_manifest.csv"
OUTPUT_DIR = "compressed_videos"
COMPRESSION_SETTINGS = {
    "original": None,
    "crf23": {"resolution": "1280:720", "crf": "23"},
    "crf28": {"resolution": "854:480", "crf": "28"},
    "crf35": {"resolution": "640:360", "crf": "35"},
}


df = pd.read_csv(MANIFEST)

for _, row in df.iterrows():

    input_video = os.path.join(DATASET_DIR, row["video_path"])

    filename = os.path.basename(row["video_path"])

    for level, settings in COMPRESSION_SETTINGS.items():

        save_folder = os.path.join(OUTPUT_DIR, level)

        os.makedirs(save_folder, exist_ok=True)

        output_video = os.path.join(save_folder, filename)

        # Original video (just copy)
        if level == "original":
            shutil.copy2(input_video, output_video)
            print(f"Copied: {filename}")
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-vf", f"scale={settings['resolution']}",
            "-c:v", "libx264",
            "-crf", settings["crf"],
            "-pix_fmt", "yuv420p",
            output_video
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"Created {level}: {filename}")

print("\nFinished!")
