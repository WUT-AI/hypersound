import math
from abc import ABC
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from hypersound.models.siren import SIREN

from .inr import INR


@dataclass
class ParamInfo:
    start_idx: int
    end_idx: int
    shape: tuple[int, ...]


def calc_fan_in_and_out(shape: tuple[int, ...]) -> tuple[int, int]:
    """Code copied from hypnettorch library"""
    assert len(shape) > 1

    fan_in = shape[1]
    fan_out = shape[0]

    if len(shape) > 2:
        receptive_field_size = int(torch.prod(torch.tensor(shape[2:])))
    else:
        receptive_field_size = 1

    fan_in *= receptive_field_size
    fan_out *= receptive_field_size

    return fan_in, fan_out


class HyperNetwork(ABC, torch.nn.Module):
    pass


class MLPHyperNetwork(HyperNetwork):
    def __init__(
        self,
        target_network: INR,
        shared_params: list[str],
        input_size: int,
        layer_sizes: list[int],
    ):
        super().__init__()
        self.input_size = input_size
        self._param_info = self._compute_param_info(target_network, shared_params)

        layers: list[nn.Linear] = []

        in_sizes = [input_size] + layer_sizes
        out_sizes = layer_sizes + [target_network.num_params(shared_params=shared_params)]

        for n_in, n_out in zip(in_sizes, out_sizes):
            layer = nn.Linear(n_in, n_out)
            layers.append(layer)

        self.net = nn.ModuleList(layers)
        self.activation_fn = nn.ELU()

        self._init_weights(target_network)

    def __call__(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        return super().__call__(x)

    def forward(self, x: Tensor) -> dict[str, Tensor]:  # type: ignore
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = self.activation_fn(x)

        return {
            param_name: x[:, info.start_idx : info.end_idx].reshape((-1, *info.shape))
            for param_name, info in self._param_info.items()
        }

    def _init_weights(self, target_network: INR, init_var: float = 1.0):
        """All comments relate to `hypnettorch.hnets.mlp_hnet`"""

        # Compute input variance - #608-617
        # Since we only have single input this variance is simply equal to initial variance
        input_variance = init_var

        # Init hidden layers #636-651
        # Ignore batch norm layers since we don't use them
        for layer in cast(list[nn.Linear], self.net)[:-1]:
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")  # type: ignore
            nn.init.zeros_(layer.bias)

        # Current variances and values from lines #652-697 default to Nones
        # Init biases of last layer to zero # 705-706
        nn.init.zeros_(cast(list[nn.Linear], self.net)[-1].bias)

        # Line #705
        c_relu = 2

        fan_in, _ = cast(tuple[int, int], nn.init._calculate_fan_in_and_fan_out(self.net[-1].weight))  # type: ignore

        """Generic initialization."""
        for i in range(target_network.n_layers):
            if f"b{i}" in self._param_info:
                c_bias = 2

                b_info = self._param_info[f"b{i}"]
                m_fan_out = b_info.shape[0]
                m_fan_in = m_fan_out
                var_in = c_relu / (2.0 * fan_in * input_variance)
                self._init_last_layer_for_param(b_info, var_in)
            else:
                c_bias = 1

            if f"w{i}" in self._param_info:
                w_info = self._param_info[f"w{i}"]
                m_fan_in, m_fan_out = calc_fan_in_and_out(w_info.shape)

                var_in = c_relu / (c_bias * m_fan_in * fan_in * input_variance)
                self._init_last_layer_for_param(w_info, var_in)

        """Pseudo-SIREN initialization"""
        tn_heads: nn.Linear = cast(list[nn.Linear], self.net)[-1]
        tn_heads_w, tn_heads_b = tn_heads.weight, tn_heads.bias

        if isinstance(target_network, SIREN):
            for i in range(target_network.n_layers):
                if f"w{i}" in self._param_info:
                    w_info = self._param_info[f"w{i}"]
                    _, n_in = w_info.shape
                    if n_in == 1:
                        bound = 1 / n_in
                    else:
                        bound = math.sqrt(6.0 / n_in)
                        if target_network.gradient_fix:
                            bound = bound / float(target_network.params[f"o{i}"])

                    with torch.no_grad():
                        tn_heads_w[w_info.start_idx : w_info.end_idx, :].multiply_(torch.tensor(1e-3))
                        tn_heads_b[w_info.start_idx : w_info.end_idx].uniform_(-bound, bound)

                    if f"b{i}" in self._param_info:
                        b_info = self._param_info[f"b{i}"]
                        with torch.no_grad():
                            tn_heads_w[b_info.start_idx : b_info.end_idx, :].multiply_(torch.tensor(1e-3))
                            tn_heads_b[b_info.start_idx : b_info.end_idx].uniform_(-bound, bound)

        """Initialization for non-layer learnable parameters"""
        with torch.no_grad():
            for param_name, param_info in self._param_info.items():
                if param_name == "freq":
                    tn_heads_w[param_info.start_idx : param_info.end_idx, :].uniform_(-1e-4, 1e-4)
                    for i, base_freq in enumerate(target_network.params[param_name].data):
                        tn_heads_b[param_info.start_idx + i : param_info.start_idx + i + 1].uniform_(
                            base_freq - 1e-2, base_freq + 1e-2
                        )
                if "o" in param_name:
                    tn_heads_w[param_info.start_idx : param_info.end_idx, :].uniform_(-1e-4, 1e-4)
                    omega = target_network.params[param_name].data
                    tn_heads_b[param_info.start_idx : param_info.end_idx].uniform_(omega - 1e-2, omega + 1e-2)

    def _init_last_layer_for_param(self, param_info: ParamInfo, var_in: float):
        var = var_in

        std = math.sqrt(var)
        a = math.sqrt(3.0) * std

        nn.init._no_grad_uniform_(  # type: ignore
            cast(list[nn.Linear], self.net)[-1].weight[param_info.start_idx : param_info.end_idx, :], -a, a
        )

    def _compute_param_info(self, target_network: INR, shared_params: list[str]) -> dict[str, ParamInfo]:
        info: dict[str, ParamInfo] = {}
        current_idx = 0
        for param_name, param in target_network.params.items():
            if param_name not in shared_params:
                info[param_name] = ParamInfo(
                    start_idx=current_idx, end_idx=current_idx + param.numel(), shape=param.shape
                )
                current_idx += param.numel()

        assert current_idx == target_network.num_params(shared_params=shared_params)

        return info

    @property
    def param_info(self) -> dict[str, ParamInfo]:
        return self._param_info
