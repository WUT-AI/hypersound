import math
from typing import Literal, Optional, Union, cast, overload

import torch
import torch.nn
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from hypersound.cfg import TargetNetworkMode


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(  # type: ignore
        self,
        x: Tensor,
    ) -> Tensor:
        return super().__call__(x)


class INR(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation_fn: nn.Module,
        bias: bool,
        mode: TargetNetworkMode,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = 1 + len(hidden_sizes)
        self.mode = mode
        self.bias = bias
        self.params: dict[str, Parameter] = {}

        self._activation_fn = activation_fn

        for i, (n_in, n_out) in enumerate(zip([input_size] + hidden_sizes, hidden_sizes + [output_size])):
            w = Parameter(torch.empty((n_out, n_in)), requires_grad=True)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))  # type: ignore
            self.params[f"w{i}"] = w

            if bias is not None:
                b = Parameter(torch.empty((n_out,)), requires_grad=True)
                fan_in, _ = cast(tuple[int, int], nn.init._calculate_fan_in_and_fan_out(w))  # type: ignore
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)
                self.params[f"b{i}"] = b

        for name, param in self.params.items():
            self.register_parameter(name, param)

    def num_params(self, shared_params: Optional[list[str]] = None) -> int:
        num_params = 0
        if shared_params is None:
            shared_params = []

        learnable_params = {
            param_name: param
            for param_name, param in self.params.items()
            if param.requires_grad and param_name not in shared_params
        }

        for param in learnable_params.values():
            num_params += param.numel()

        return num_params

    def freeze_params(self, shared_params: list[str]) -> None:  # TODO: Verify
        assert self.mode is TargetNetworkMode.TARGET_NETWORK
        for param_name, param in self.params.items():
            if param_name not in shared_params:
                param.requires_grad = False

    @overload
    def __call__(
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        *,
        return_activations: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def __call__(
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        *,
        return_activations: Literal[True],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        ...

    def __call__(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        *,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        return super().__call__(x, weights, return_activations=return_activations)

    def forward(  # type: ignore
        self,
        x: Tensor,
        weights: Optional[dict[str, Tensor]] = None,
        *,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        """
        Forward function for INR network. Works in two modes: if `weights` is None, behaves as a
        standard MLP, expects input size [S, I] and returns output [S, O], where S is sequence
        length and I and O are input and output sizes.
        If `weights` is provided will run batched inference with multiple models in parallel,
        with expected input shape of [N, S, I] and output shape [N, S, O]. Weights are expected to
        be dict of Tensors, where first dimension of each tensor is always N.
        """
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

    def _forward(self, x: Tensor, layer_idx: int, weights: Optional[dict[str, Tensor]] = None) -> Tensor:
        weight_name = f"w{layer_idx}"
        bias_name = f"b{layer_idx}"

        weight_matrix, bias = None, None

        if self.mode is TargetNetworkMode.INR:
            if weights is not None:
                raise ValueError("Can't provide weights in `inr` mode.")

            x = torch.matmul(x, self.params[weight_name].T)
            if self.bias:
                x = x + self.params[bias_name]

        elif self.mode in (TargetNetworkMode.TARGET_NETWORK, TargetNetworkMode.RESIDUAL, TargetNetworkMode.MODULATOR):
            if weights is None:
                raise ValueError("`weights` are required for inference in `target_network` mode.")

            if self.mode in (TargetNetworkMode.RESIDUAL, TargetNetworkMode.MODULATOR):
                weight_matrix = weights[weight_name]
                if not self.params[weight_name].requires_grad or not (
                    self.bias and self.params[bias_name].requires_grad  # FIXME?
                ):
                    raise ValueError()
                if self.bias:
                    bias = weights[bias_name]

                if self.mode is TargetNetworkMode.RESIDUAL:
                    weight_matrix = weight_matrix + torch.stack(
                        x.shape[0] * [cast(Tensor, self.params[weight_name])], dim=0
                    )
                else:
                    weight_matrix = weight_matrix * torch.stack(
                        x.shape[0] * [cast(Tensor, self.params[weight_name])], dim=0
                    )

            else:
                weight_matrix = weights.get(
                    weight_name, torch.stack(x.shape[0] * [cast(Tensor, self.params[weight_name])], dim=0)
                )

                if self.bias:
                    bias = weights.get(
                        bias_name, torch.stack(x.shape[0] * [cast(Tensor, self.params[bias_name])], dim=0)
                    )

            assert weight_matrix is not None
            x = torch.bmm(x, weight_matrix.permute(0, 2, 1))
            if self.bias:
                assert bias is not None
                if self.mode is TargetNetworkMode.RESIDUAL:
                    x = x + bias.unsqueeze(1)
                elif self.mode is TargetNetworkMode.MODULATOR:
                    x = x * bias.unsqueeze(1)
                else:
                    x = x + bias.unsqueeze(1)
        else:
            raise ValueError(f"Unknown mode: `{self.mode}.")

        return x
