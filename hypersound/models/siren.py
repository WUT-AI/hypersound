from typing import Optional, Union, cast

import torch
import torch.nn
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from hypersound.cfg import TargetNetworkMode
from hypersound.models.meta.inr import INR


class Sine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return torch.sin(x)


class SIREN(INR):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        bias: bool,
        mode: TargetNetworkMode,
        omega_0: float,
        omega_i: float,
        learnable_omega: bool,
        gradient_fix: bool,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=bias,
            activation_fn=Sine(),
            mode=mode,
        )

        for i in range(self.n_layers):
            omega_val = torch.ones((1,), dtype=torch.float32)
            if i == 0:
                omega_val *= omega_0
            else:
                omega_val *= omega_i
            omega = Parameter(omega_val, requires_grad=learnable_omega)

            self.params[f"o{i}"] = omega
            self.register_parameter(f"o{i}", omega)

        self.gradient_fix = gradient_fix
        self.init_siren()

    def init_siren(self) -> None:
        for i in range(self.n_layers):
            w_i = self.params[f"w{i}"]
            _, n_features_in = w_i.shape

            if i == 0:
                std = 1 / n_features_in
            else:
                # NOTE: Version used in https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb  # noqa
                std = (6.0 / n_features_in) ** 0.5
                if self.gradient_fix:
                    std /= float(self.params["o0"])

            with torch.no_grad():
                w_i.uniform_(-std, std)

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:

        activations: list[tuple[Tensor, Tensor]] = []

        for i in range(self.n_layers):
            if weights is None:
                omega = cast(Tensor, self.params[f"o{i}"])
                x = x * omega
            else:
                omega = weights.get(f"o{i}", torch.stack(x.shape[0] * [cast(Tensor, self.params[f"o{i}"])], dim=0))
                x = x * omega.unsqueeze(-1)

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
