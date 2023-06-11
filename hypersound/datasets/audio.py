from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

IndicesAudioAndSpectrogram = tuple[Tensor, Tensor, Tensor]


class AudioDataset(Dataset[IndicesAudioAndSpectrogram]):
    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, int]],
        duration: float,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        data_norm: bool,
        index_scaling: Optional[tuple[float, float]],
        proportional_index_scaling: bool,
    ):
        self._dataset = dataset
        self.duration = duration
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.data_norm = data_norm

        self.total_samples = int(duration * sample_rate)
        self.indices = torch.arange(0, self.total_samples, dtype=torch.float).unsqueeze(-1)

        if index_scaling:
            min_val, max_val = index_scaling

            if proportional_index_scaling:
                # Normalize config setting to 1 second
                min_val *= self.duration
                max_val *= self.duration

            assert min_val < max_val
            self.indices = min_val + (max_val - min_val) * self.indices / (self.total_samples - 1)

        self.spec_transform = nn.ModuleList(
            [
                MelSpectrogram(sample_rate=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
                AmplitudeToDB(),
            ]
        )

    @property
    def shape(self):
        n_channels = 1
        freq_size = self.n_mels

        return (
            (self.total_samples, 1),
            (self.total_samples,),
            (n_channels, freq_size, 1 + self.total_samples // self.hop_length),
        )

    def to_spectrogram(self, y: Tensor) -> Tensor:
        for transform in self.spec_transform:
            y = transform(y)
        y = y.unsqueeze(-3)
        return y

    def __getitem__(self, idx: int) -> IndicesAudioAndSpectrogram:
        # Load raw recording from the underlying dataset
        y = self._dataset[idx][0]
        samplerate = self._dataset[idx][1]

        y = y[0]  # type: ignore
        assert samplerate == self.sample_rate, f"Got {samplerate}, expected {self.sample_rate}"
        assert y.dim() == 1  # type: ignore
        assert len(y) == self.total_samples, "Audio signal has an invalid length"

        # Normalize audio
        if self.data_norm and y.abs().max() > 0:
            y = y / y.abs().max()  # type: ignore

        # Generate spectrogram
        spec = self.to_spectrogram(y)  # type: ignore

        # Generate time indices
        indices = self.indices

        return indices, y, spec  # type: ignore

    def __len__(self):
        return len(self._dataset)  # type: ignore

    def __str__(self) -> str:
        return (
            f"<AudioDataset ({self.duration:.2f} s @ {self.sample_rate} Hz) "
            f"[n_mels: {self.n_mels}, n_fft: {self.n_fft}, hop_length: {self.hop_length}] "
            f"=> {self.shape}>"
        )
