from typing import Sequence

import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torchvision import disable_beta_transforms_warning, os, transforms

disable_beta_transforms_warning()
from torchvision.transforms.v2 import (CenterCrop, Compose, Lambda, Normalize,
                                       RandomCrop, RandomHorizontalFlip,
                                       ToImageTensor)

from .dataset import VideoClassificationDataset
from .transforms import RandomShortSideScale


class VideoClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        num_classes: int,
        num_frames: int = 8,
        frame_stride_rate: int = 2,
        size: int = 224,
        min_scale: int = 256,
        max_scale: int = 320,
        flip_prob: float = 0.5,
        mean: Sequence = (0.45, 0.45, 0.45),
        std: Sequence = (0.225, 0.225, 0.225),
        batch_size: int = 2,
        workers: int = 4,
        decord_treads: int = 1,
    ):
        """Video Classification Datamodule

        Args:
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            num_classes: Number of target classes
            num_frames: Number of frames in a sampled clip
            frame_stride_rate: Temporal stride rate when sampling clip frames
            size: Crop size
            min_scale: Minimum size of random shortest side scale
            max_scale: Maximum size of random shortest side scale
            flip_prob: Probability of applying horizontal flip
            mean: Normalization means
            std: Normalization standard deviations
            batch_size: Number of batch samples
            workers: Number of data loader workers
            decord_treads: Number of threads used by Decord
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_stride_rate = frame_stride_rate
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.flip_prob = flip_prob
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.workers = workers
        self.decord_treads = decord_treads

        self.transforms_train = Compose(
            [
                ToImageTensor(),
                Rearrange("t h w c -> t c h w"),
                Lambda(lambda x: x / 255.0),
                RandomHorizontalFlip(self.flip_prob),
                RandomShortSideScale(self.min_scale, self.max_scale),
                RandomCrop(self.size),
                Normalize(mean=self.mean, std=self.std),
                Rearrange("t c h w -> c t h w"),
            ]
        )

        self.transforms_test = transforms.Compose(
            [
                ToImageTensor(),
                Rearrange("t h w c -> t c h w"),
                Lambda(lambda x: x / 255.0),
                RandomShortSideScale(self.size, self.size),
                CenterCrop(self.size),
                Normalize(mean=self.mean, std=self.std),
                Rearrange("t c h w -> c t h w"),
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = VideoClassificationDataset(
                os.path.join(self.root, "train"),
                num_frames=self.num_frames,
                frame_stride_rate=self.frame_stride_rate,
                clip_sample_mode="random",
                transforms=self.transforms_train,
                threads=self.decord_treads,
            )
            self.val_dataset = VideoClassificationDataset(
                os.path.join(self.root, "val"),
                num_frames=self.num_frames,
                frame_stride_rate=self.frame_stride_rate,
                clip_sample_mode="center",
                transforms=self.transforms_test,
                threads=self.decord_treads,
            )
        elif stage == "test":
            self.test_dataset = VideoClassificationDataset(
                os.path.join(self.root, "test"),
                num_frames=self.num_frames,
                frame_stride_rate=self.frame_stride_rate,
                clip_sample_mode="center",
                transforms=self.transforms_test,
                threads=self.decord_treads,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )
