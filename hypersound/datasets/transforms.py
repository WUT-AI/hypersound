from typing import Callable

import torch
from scipy.signal import lfilter  # type: ignore
from torch import Tensor


class Transform:
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RandomApply(Transform):
    def __init__(self, transform: Callable[[Tensor], Tensor], p: float = 0.5):
        assert 0.0 <= p <= 1.0
        self.transform = transform
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            x = self.transform(x)
        return x


class RandomCrop(Transform):
    def __init__(self, total_samples: int, random: bool = True):
        self.total_samples = total_samples
        self.random = random

    def __call__(self, x: Tensor) -> Tensor:
        if not self.random:
            start = 0
        else:
            start = int(torch.randint(x.shape[-1] - self.total_samples, size=(1,)).item())

        x = x[..., start : start + self.total_samples]
        return x


class Dequantize(Transform):
    def __init__(self, bit_depth: int):
        self.bit_depth = bit_depth

    def __call__(self, x: Tensor) -> Tensor:
        x = x + (torch.rand(x.shape[-1]) - 0.5) / 2**self.bit_depth
        return x


class RandomPhaseMangle(Transform):
    def __init__(self, min_f: int, max_f: int, amplitude: float, sample_rate: int):
        self.min_f = torch.tensor(min_f)
        self.max_f = torch.tensor(max_f)
        self.amplitude = torch.tensor(amplitude)
        self.sample_rate = sample_rate

    def random_angle(self) -> Tensor:
        min_f = torch.log(self.min_f)
        max_f = torch.log(self.max_f)

        rand = torch.exp(torch.rand(1).item() * (max_f - min_f) + min_f)
        rand = 2 * torch.pi * rand / self.sample_rate

        return rand

    def pole_to_z_filter(self, angle: Tensor) -> tuple[list[float], list[float]]:
        z0 = self.amplitude * torch.exp(1j * angle)
        a = [1.0, float(-2 * torch.real(z0)), float(torch.abs(z0) ** 2)]
        b = [float(torch.abs(z0) ** 2), float(-2 * torch.real(z0)), 1.0]
        return b, a

    def __call__(self, x: Tensor) -> Tensor:
        angle = self.random_angle()
        b, a = self.pole_to_z_filter(angle)

        return torch.tensor(lfilter(b, a, x.numpy())).float()
