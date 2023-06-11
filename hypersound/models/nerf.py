from typing import Optional, Union, cast

import torch
import torch.nn
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from hypersound.cfg import TargetNetworkMode
from hypersound.models.meta.inr import INR


class NERF(INR):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        bias: bool,
        mode: TargetNetworkMode,
        encoding_length: int,
        learnable_encoding: bool,
    ):
        super().__init__(
            input_size=2 * encoding_length * input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=bias,
            activation_fn=nn.ReLU(),
            mode=mode,
        )
        self.input_size = input_size  # override super().input_size
        self.encoding_length = encoding_length

        freq = torch.ones((encoding_length,), dtype=torch.float32)
        for i in range(len(freq)):
            freq[i] = 2**i
        freq = freq * torch.pi
        freq = Parameter(freq, requires_grad=learnable_encoding)

        self.params["freq"] = freq
        self.register_parameter("freq", freq)

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        if weights is None:
            # Single mode, x --> (num_samples, input_size)
            freq = cast(Tensor, self.params["freq"])  # (encoding_length,)
        else:
            # Batch mode, x --> (batch_size, num_samples, input_size)
            freq = weights.get("freq", cast(Tensor, self.params["freq"]).tile(x.shape[0], 1))
            freq = freq.unsqueeze(1).unsqueeze(1)  # (batch_size, _, _, encoding_length)

        x = x.unsqueeze(-1) * freq
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1).flatten(-2, -1)
        # x --> (_batch_size_, num_samples, input_size * encoding_length * 2)

        activations: list[tuple[Tensor, Tensor]] = []

        for i in range(self.n_layers):
            x = self._forward(x, layer_idx=i, weights=weights)

            h = x

            if i != self.n_layers - 1:
                x = self._activation_fn(x)

            if return_activations:
                activations.append((x, h))

        if return_activations:
            return x, activations
        else:
            return x
