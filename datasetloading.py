import os
import subprocess

# List of categories
CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
              "Fighting", "RoadAccidents", "Robbery", "Shooting",
              "Shoplifting", "Stealing", "Vandalism"]

# Base directory containing video categories
BASE_DIR = "/content/data/DCSASS Dataset/"

# Loop through each category
for category in CATEGORIES:
    category_path = os.path.join(BASE_DIR, category)

    if not os.path.isdir(category_path):
        print(f"❌ Category folder not found: {category}")
        continue

    print(f"🔹 Processing category: {category}")

    # Loop through each video folder in the category
    for video_folder in sorted(os.listdir(category_path)):
        video_folder_path = os.path.join(category_path, video_folder)

        # Ensure it's a folder and contains sub-videos
        if not os.path.isdir(video_folder_path):
            continue

        # Output merged video path
        merged_video_path = os.path.join(category_path, f"{video_folder}_merged.mp4")

        # Get all clips inside this video folder
        video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp4")])

        if len(video_files) == 0:
            print(f"⚠️ No video clips found in: {video_folder_path}")
            continue

        # Create a file listing all video clips
        merge_list_path = os.path.join(video_folder_path, "merge_list.txt")
        with open(merge_list_path, "w") as f:
            for video in video_files:
                f.write(f"file '{os.path.join(video_folder_path, video)}'\n")

        # Run FFmpeg command to merge videos
        ffmpeg_cmd = f"ffmpeg -f concat -safe 0 -i '{merge_list_path}' -c copy '{merged_video_path}'"
        subprocess.run(ffmpeg_cmd, shell=True)

        print(f"✅ Merged video saved: {merged_video_path}")

print("🎉 All video clips merged successfully!")
