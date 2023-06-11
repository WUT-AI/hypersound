from math import ceil
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# https://github.com/wesbz/SoundStream/blob/main/net.py


# Signal encoder
# --------------


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)  # type: ignore
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: Tensor):  # type: ignore
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, adjust_padding: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)  # type: ignore
        self.adjust_padding = adjust_padding
        self.causal_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
            - self.stride[0]
            + self.adjust_padding
        )

    def forward(self, x: Tensor, output_size: Optional[list[int]] = None):  # type: ignore
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation  # type: ignore
        )
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )[..., : -self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, layer_norm_size: Optional[int] = None):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            # SEANet: first kernel_size=3
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation),
            nn.LayerNorm([out_channels, layer_norm_size]) if layer_norm_size else nn.Identity(),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.LayerNorm([out_channels, layer_norm_size]) if layer_norm_size else nn.Identity(),
            nn.ELU(),
        )

    def forward(self, x: Tensor):  # type: ignore
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, stride: int, layer_norm_size: Optional[int] = None):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=channels // 2, out_channels=channels // 2, dilation=1, layer_norm_size=layer_norm_size
            ),
            ResidualUnit(
                in_channels=channels // 2, out_channels=channels // 2, dilation=3, layer_norm_size=layer_norm_size
            ),
            ResidualUnit(
                in_channels=channels // 2, out_channels=channels // 2, dilation=9, layer_norm_size=layer_norm_size
            ),
            CausalConv1d(in_channels=channels // 2, out_channels=channels, kernel_size=2 * stride, stride=stride),
            nn.LayerNorm([channels, ceil(layer_norm_size / stride)]) if layer_norm_size else nn.Identity(),
            nn.ELU(),
        )

    def forward(self, x: Tensor):  # type: ignore
        return self.layers(x)


class Encoder(nn.Module):
    """Audio encoder."""

    def __init__(
        self,
        C: int,  # SEANET: 32
        D: int,
        layer_norm: bool,
        **kwargs: Any,
    ):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),  #
            nn.LayerNorm([16, 32768]) if layer_norm else nn.Identity(),  # 16x32768
            nn.ELU(),
            EncoderBlock(
                channels=2 * C, stride=2, layer_norm_size=32768 if layer_norm else None
            ),  # SEANet: Stride 2  # AudioGen 2
            EncoderBlock(
                channels=4 * C, stride=4, layer_norm_size=16384 if layer_norm else None
            ),  # SEANet: Stride 2  # AudioGen 2
            EncoderBlock(
                channels=8 * C, stride=5, layer_norm_size=4096 if layer_norm else None
            ),  # SEANet: Stride 8  # AudioGen 2
            EncoderBlock(
                channels=16 * C, stride=8, layer_norm_size=820 if layer_norm else None
            ),  # SEANet: Stride 8  # AudioGen 4
            CausalConv1d(
                in_channels=16 * C, out_channels=D, kernel_size=3
            ),  # SEANet: kernel_size=7, channels=128  # AudioGen: 7
        )

    def __call__(self, x: Tensor) -> Tensor:  # type: ignore
        return super().__call__(x)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = self.net(x.unsqueeze(1))
        return x.flatten(1, 2)

    def output_width(self, input_length: Any) -> int:
        return ceil(input_length / (2 * 4 * 5 * 8))


# STFT discriminator
# ----------------------


class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels: int, N: int, m: int, s_t: int, s_f: int):
        super().__init__()

        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=N, kernel_size=(3, 3), padding="same"),
            nn.ELU(),
            nn.Conv2d(in_channels=N, out_channels=m * N, kernel_size=(s_f + 2, s_t + 2), stride=(s_f, s_t)),
        )

        self.skip_connection = nn.Conv2d(
            in_channels=in_channels, out_channels=m * N, kernel_size=(1, 1), stride=(s_f, s_t)
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.layers(F.pad(x, [self.s_t + 1, 0, self.s_f + 1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    """STFT discriminator."""

    def __init__(
        self,
        C: int,
        win_length: int,
        hop_size: int,
        **kwargs: Any,
    ):
        super().__init__()

        self.win_length = win_length
        self.hop_size = hop_size

        self.window: Tensor
        self.register_buffer("window", torch.hann_window(self.win_length))

        frequency_bins = self.win_length // 2

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=32, N=C, m=2, s_t=1, s_f=2), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=2 * C, N=2 * C, m=2, s_t=2, s_f=2), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=4 * C, N=4 * C, m=1, s_t=1, s_f=2), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=4 * C, N=4 * C, m=2, s_t=2, s_f=2), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=8 * C, N=8 * C, m=1, s_t=1, s_f=2), nn.ELU()),
                nn.Sequential(ResidualUnit2d(in_channels=8 * C, N=8 * C, m=2, s_t=2, s_f=2), nn.ELU()),
                nn.Conv2d(in_channels=16 * C, out_channels=1, kernel_size=(frequency_bins // 2**6, 1)),
            ]
        )

    def __call__(self, x: Tensor) -> Tensor:  # type: ignore
        return super().__call__(x)

    # def features_lengths(self, lengths):
    #     return [
    #         lengths-6,
    #         lengths-6,
    #         torch.div(lengths-5, 2, rounding_mode="floor"),
    #         torch.div(lengths-5, 2, rounding_mode="floor"),
    #         torch.div(lengths-3, 4, rounding_mode="floor"),
    #         torch.div(lengths-3, 4, rounding_mode="floor"),
    #         torch.div(lengths+1, 8, rounding_mode="floor"),
    #         torch.div(lengths+1, 8, rounding_mode="floor")
    #     ]

    def forward(self, x: Tensor) -> list[Tensor]:  # type: ignore

        x = torch.stft(
            x,
            self.win_length,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=False,
        ).permute(0, 3, 1, 2)

        feature_map: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
            # print(x.shape)  # DEBUG
        return feature_map
