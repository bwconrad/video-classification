"""
Download a subset from the Kinetics 400/600/700 dataset
"""
import argparse
import csv
import os
import random
import subprocess
from itertools import islice


def download_dataset(
    dataset_file: str,
    output_dir: str,
    num_classes: int,
    num_videos_per_class: int,
    seed: int | None,
):
    # Set the random seed
    random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read the dataset csv
    with open(dataset_file, "r") as file:
        reader = csv.DictReader(file)
        videos = list(reader)

    # Group videos by label
    videos_by_label = {}
    for video in videos:
        label = video["label"]
        if label not in videos_by_label:
            videos_by_label[label] = []
        videos_by_label[label].append(video)

    # Sort dictionary labels so that the first num_classes are always the same
    videos_by_label = dict(sorted(videos_by_label.items()))

    # Select the first num_classes classes
    videos_by_label = dict(islice(videos_by_label.items(), num_classes))

    # Shuffle the videos randomly within each label
    for label, label_videos in videos_by_label.items():
        random.shuffle(label_videos)

    # Check that all classes have num_videos_per_class
    min_videos_per_class = min(len(videos) for videos in videos_by_label.values())
    assert num_videos_per_class <= min_videos_per_class

    # Download the subset of videos
    for label, label_videos in videos_by_label.items():
        # Create a subdirectory for the class
        class_output_dir = os.path.join(output_dir, label)
        os.makedirs(class_output_dir, exist_ok=True)

        downloaded_count = 0
        i = 0
        while downloaded_count < num_videos_per_class and i < len(label_videos):
            video = label_videos[i]
            video_id = video["youtube_id"]
            time_start = video["time_start"]
            time_end = video["time_end"]

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            output_path = os.path.join(class_output_dir, f"{video_id}.mp4")

            try:
                # Use yt-dlp to download the video segment
                subprocess.run(
                    [
                        "yt-dlp",
                        "-f",
                        "mp4",
                        "-o",
                        output_path,
                        "--download-sections",
                        f"*{time_start}-{time_end}",
                        video_url,
                    ],
                    check=True,
                )

                print(
                    f'Downloaded video {downloaded_count+1}/{num_videos_per_class} for class "{label}": {video_id}'
                )
                downloaded_count += 1
            except subprocess.CalledProcessError:
                print(f'Failed to download video for class "{label}": {video_id}')
                i += 1
                continue

            i += 1

    print("Download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download a subset of the Kinetics dataset."
    )
    parser.add_argument(
        "--dataset-file", "-i", type=str, help="Path to the dataset file in CSV format."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for downloaded videos."
    )
    parser.add_argument(
        "--n-per-class", "-n", type=int, help="Number of videos to download per class."
    )
    parser.add_argument(
        "--n-class", "-c", type=int, help="Number of videos classes to download."
    )
    parser.add_argument("--seed", "-s", type=int, default=12345, help="Random seed.")

    args = parser.parse_args()

    download_dataset(
        args.dataset_file, args.output_dir, args.n_class, args.n_per_class, args.seed
    )


if __name__ == "__main__":
    main()
