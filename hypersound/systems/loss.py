import librosa.filters
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


class MultiSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes: list[int],
        hop_sizes: list[int],
        win_lengths: list[int],
        sample_rate: int,
        n_bins: int,
        freq_weights_p: float = 0.0,
        freq_weights_warmup_epochs: int = 500,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fft_size,
                    hop_size,
                    win_length,
                    sample_rate,
                    n_bins,
                    freq_weights_p=freq_weights_p,
                    freq_weights_warmup_epochs=freq_weights_warmup_epochs,
                )
            ]

    def update(self, current_epoch) -> None:
        for loss in self.stft_losses:
            loss.update(current_epoch)

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:  # type: ignore
        return torch.stack([loss(x, x_hat) for loss in self.stft_losses]).mean()


class STFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
        sample_rate: int,
        n_bins: int,
        freq_weights_p: float,
        freq_weights_warmup_epochs: int,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.sample_rate = sample_rate
        self.n_bins = n_bins

        self.eps = 1e-8

        assert sample_rate is not None  # Must set sample rate to use mel scale
        # assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
        fb: npt.NDArray[np.float32] = librosa.filters.mel(
            sr=sample_rate, n_fft=fft_size, n_mels=n_bins
        )  # type: ignore  # noqa

        self.fb: Tensor
        self.register_buffer(
            "fb",
            torch.tensor(fb).unsqueeze(0),
        )

        self.freq_weights_p = freq_weights_p
        self.freq_weights_warmup_epochs = freq_weights_warmup_epochs
        self.mask = self.compute_mask(0)

    def update(self, current_epoch) -> None:
        self.mask = self.compute_mask(current_epoch)

    def compute_mask(self, current_epoch: int) -> Tensor:
        if self.freq_weights_p == 0:
            return torch.ones((self.n_bins, 1))
        else:
            mask = (torch.arange(128) + 1) ** self.freq_weights_p

        if current_epoch > self.freq_weights_warmup_epochs:
            return self.mask

        else:
            mask = mask / mask.sum() * self.n_bins  # Make mask sum to the same value as mask of ones
            alpha = 1 - current_epoch / self.freq_weights_warmup_epochs
            mask = alpha * torch.ones(mask.shape) + (1 - alpha) * mask
            return mask.unsqueeze(1)

    def stft(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps))
        x_phs = torch.angle(x_stft)

        return x_mag, x_phs

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:  # type: ignore
        self.window = self.window.to(x.device)
        x_mag, _ = self.stft(x.view(-1, x.size(-1)))
        x_hat_mag, _ = self.stft(x_hat.view(-1, x_hat.size(-1)))

        x_mag = torch.matmul(self.fb, x_mag)
        x_hat_mag = torch.matmul(self.fb, x_hat_mag)

        # Standardize?
        # x_mag = (x_mag - x_mag.mean([1, 2], keepdim=True)) / x_mag.std([1, 2], keepdim=True)
        # x_hat_mag = (x_hat_mag - x_hat_mag.mean([1, 2], keepdim=True)) / x_hat_mag.std([1, 2], keepdim=True)

        # compute loss terms
        l1_corrected = (x_mag - x_hat_mag).abs() * self.mask.to(x.device)
        l1 = l1_corrected.mean()
        l2_corrected = (x_mag - x_hat_mag).pow(2) * self.mask.to(x.device)
        l2 = l2_corrected.mean().sqrt()

        return l1 + l2
