import os
from typing import Callable, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset
from torchvision import disable_beta_transforms_warning
from torchvision.io import read_video

disable_beta_transforms_warning()
from torchvision import datapoints


class VideoClassificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        backend: str = "decord",
        num_clips: int = 1,
        num_frames: int = 8,
        frame_stride_rate: int = 2,
        clip_sample_mode: str = "random",
        transforms: Callable | None = None,
        threads: int = 1,
    ):
        self.root = root
        self.backend = backend
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_stride_rate = frame_stride_rate
        self.clip_sample_mode = clip_sample_mode
        self.transform = transforms
        self.threads = threads

        assert self.clip_sample_mode in ["random", "center"]

        # Load paths and labels
        self.class_to_idx = self.find_classes()
        self.paths, self.labels = self.find_samples()
        assert len(self.paths) == len(self.labels)
        print(f"Loaded {len(self.paths)} videos from {self.root}")

        self.num_classes = len(self.class_to_idx)

    def find_classes(self) -> dict:
        """
        Creates a dictionary mapping class names to class indices.

        Returns:
            dict: Dictionary mapping class names to class indices.
        """
        class_names = sorted(os.listdir(self.root))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        return class_to_idx

    def sample_random_clip(self, video_length: int) -> tuple[int, int]:
        """
        Randomly chooses the start and end indices of a clip from a video

        Args:
            video_length (int): Number of frames of the whole video

        Returns:
            tuple: Tuple of the start and end indices
        """
        clip_length = self.num_frames * self.frame_stride_rate
        end = np.random.randint(clip_length, video_length)
        start = end - clip_length
        return start, end

    def sample_center_clip(self, video_length: int) -> tuple[int, int]:
        """
        Chooses the start and end indices of the center clip from a video

        Args:
            video_length (int): Number of frames of the whole video

        Returns:
            tuple: Tuple of the start and end indices
        """
        clip_length = self.num_frames * self.frame_stride_rate
        end = (video_length // 2) + (clip_length // 2)
        start = end - clip_length
        return start, end

    def find_samples(self) -> tuple[list, list]:
        """
        Creates lists of the file paths and corresponding class labels for all samples in a directory.

        Returns:
            tuple: Tuple containing lists of the file paths and class labels.
        """
        paths = []
        labels = []

        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(self.root, class_name)
            file_paths = os.listdir(class_dir)
            paths.extend(
                [os.path.join(class_dir, file_path) for file_path in file_paths]
            )
            labels.extend([self.class_to_idx[class_name]] * len(file_paths))

        return paths, labels

    def load_video_decord(self, path) -> np.ndarray:
        """
        Load video frames using the decord library.

        Args:
            path (str): Path to the video file.

        Returns:
            np.ndarray: An array containing video frames with shape (num_frames, H, W, C),
                          where 'num_frames' is the number of frames sampled from the video,
                          'C' is the number of channels (usually 3 for RGB), 'H' is the height,
                          and 'W' is the width.
        """
        # Initialize video reader
        vr = VideoReader(path, num_threads=self.threads, ctx=cpu(0))
        video_length = len(vr)

        # Sample clip frames
        if self.clip_sample_mode == "random":
            start, end = self.sample_random_clip(video_length)
        else:
            start, end = self.sample_center_clip(video_length)

        indices = np.linspace(start, end, num=self.num_frames)
        indices = np.clip(indices, start, end - 1).astype(np.int64)

        # Grab frames from video
        vr.seek(0)
        frames = vr.get_batch(indices).asnumpy()

        return frames

    def load_video_torch(self, path) -> torch.Tensor:
        """
        Load video frames using the torchvision library.

        Args:
            path (str): Path to the video file.

        Returns:
            torch.Tensor: A tensor containing video frames with shape (num_frames, C, H, W),
                          where 'num_frames' is the number of frames sampled from the video,
                          'C' is the number of channels (usually 3 for RGB), 'H' is the height,
                          and 'W' is the width.
        """
        # Load all video frames
        video, _, _ = read_video(path, output_format="TCHW", pts_unit="sec")
        video_length = len(video)

        # Sample clip start and end indices
        if self.clip_sample_mode == "random":
            start, end = self.sample_random_clip(video_length)
        else:
            start, end = self.sample_center_clip(video_length)

        # Sample all clip indices
        indices = np.linspace(start, end, num=self.num_frames)
        indices = np.clip(indices, start, end - 1).astype(np.int64)

        # Get frames from video
        frames = video[indices]

        return frames

    def __getitem__(self, index) -> dict:
        # Choose a sample
        video_path = self.paths[index]
        label = self.labels[index]

        # Load clip sample
        if self.backend == "torch":
            frames = datapoints.Video(self.load_video_torch(video_path))
        else:
            frames = datapoints.Video(self.load_video_decord(video_path))

        # Apply transformations if specified
        if self.transform:
            frames = self.transform(frames)

        return {"video": frames, "label": label, "path": video_path}

    def __len__(self) -> int:
        return len(self.paths)


if __name__ == "__main__":
    from torchvision.transforms.v2 import (Compose, Lambda, Normalize,
                                           RandomCrop, RandomHorizontalFlip,
                                           ToImageTensor)
    from transforms import RandomShortSideScale

    mean: Sequence = (0.45, 0.45, 0.45)
    std: Sequence = (0.225, 0.225, 0.225)
    t = Compose(
        [
            ToImageTensor(),
            Rearrange("t h w c -> t c h w"),
            Lambda(lambda x: x / 255.0),
            RandomHorizontalFlip(),
            RandomShortSideScale(256, 320, antialias=True),
            RandomCrop(224),
            Normalize(mean=mean, std=std),
            Rearrange("t c h w -> c t h w"),
        ]
    )
    d = VideoClassificationDataset(
        root="data/k-40/train",
        transforms=t,
        frame_stride_rate=8,
        clip_sample_mode="random",
        backend="decord",
    )
    # print(d.num_classes)
    out = d[10]
    x, y = out["video"], out["label"]
    print(x.size())
    print(x.min())
    print(x.max())
    # from torchvision.utils import save_image

    # for i, im in enumerate(x):
    #     save_image(im, f"{i}.jpg", normalize=True)
