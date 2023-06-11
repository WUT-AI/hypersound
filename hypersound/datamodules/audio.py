from typing import Any, Optional, cast

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataset import Subset

from ..datasets.audio import AudioDataset, IndicesAudioAndSpectrogram


class AudioDataModule(LightningDataModule):
    """A data module for generic audio datasets.

    Uses the `hypersound.datasets.audio.AudioDataset` wrapper to return
    pairs of (waveform, melspectrogram) tensors.

    """

    def __init__(
        self,
        train_dataset: Dataset[Any],
        validation_dataset: Dataset[Any],
        batch_size: int,
        num_workers: int,
        samples_per_epoch: int,
        file_limit: Optional[int] = None,
        file_limit_validation: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore

        self.train = AudioDataset(train_dataset, **kwargs)
        self.validation = AudioDataset(validation_dataset, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.file_limit = file_limit
        self.file_limit_validation = file_limit_validation
        self.samples_per_epoch = samples_per_epoch

    def prepare_data(self, *args: None, **kwargs: None):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.file_limit:
            _indices: list[int] = torch.randperm(len(self.train), generator=torch.Generator().manual_seed(0)).tolist()  # type: ignore  # noqa
            self.train = cast(AudioDataset, Subset(self.train, _indices[: self.file_limit]))

        if self.file_limit_validation:
            _indices: list[int] = torch.randperm(len(self.validation), generator=torch.Generator().manual_seed(0)).tolist()  # type: ignore  # noqa
            self.validation = cast(AudioDataset, Subset(self.validation, _indices[: self.file_limit_validation]))

    @property
    def shape(self):
        return self.train.shape

    def train_dataloader(self, *args: None, **kwargs: None) -> DataLoader[IndicesAudioAndSpectrogram]:
        assert self.train
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=RandomSampler(self.train, num_samples=self.samples_per_epoch),
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self, *args: None, **kwargs: None) -> DataLoader[IndicesAudioAndSpectrogram]:
        assert self.validation
        return DataLoader(
            self.validation, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True
        )

    def __repr__(self) -> str:
        return (
            f"<AudioDataModule: ["
            f'train: {len(self.train) if self.train else "---"}, '
            f'validation: {len(self.validation) if self.validation else "---"}, '
            f"] --> {self.train}, {self.validation}>"
        )
