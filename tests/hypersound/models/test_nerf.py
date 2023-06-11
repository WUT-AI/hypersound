import pytest
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor

from hypersound.cfg import TargetNetworkMode
from hypersound.models.nerf import NERF

EPS = 1e-3


@pytest.mark.parametrize(
    "num_samples, encoding_length",
    [
        (1, 1),
        (1, 6),
        (50, 5),
        (32768, 1),
        (32768, 10),
        (32768, 16),
    ],
)
def test_nerf_standard_inference(num_samples: int, encoding_length: int) -> None:
    model = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.INR,
        encoding_length=encoding_length,
        learnable_encoding=False,
    )

    x: Tensor = torch.rand((num_samples, 1))

    assert model(x, weights=None).shape == (num_samples, 1)


@pytest.mark.parametrize(
    "num_samples, encoding_length",
    [
        (1, 1),
        (1, 6),
        (50, 5),
        (32768, 1),
        (32768, 10),
        (32768, 16),
    ],
)
def test_nerf_train_loop(num_samples: int, encoding_length: int) -> None:
    model = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.INR,
        encoding_length=encoding_length,
        learnable_encoding=False,
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.L1Loss()
    x = torch.rand((num_samples, 1))
    y = torch.rand((num_samples, 1))
    y0 = model(x)

    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    assert F.mse_loss(y.squeeze(), y0.squeeze()) > EPS


@pytest.mark.parametrize(
    "num_samples, encoding_length",
    [
        (1, 1),
        (1, 6),
        (50, 5),
        (32768, 1),
        (32768, 10),
        (32768, 16),
    ],
)
def test_nerf_inr_tn_equivalence(num_samples: int, encoding_length: int) -> None:
    x = torch.rand((num_samples, 1))

    inr = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.INR,
        encoding_length=encoding_length,
        learnable_encoding=False,
    )
    y_inr = inr(x)

    tn = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
        encoding_length=encoding_length,
        learnable_encoding=False,
    )
    weights = {name: param.unsqueeze(0) for name, param in inr.params.items() if param.requires_grad}
    y_tn = tn(x.unsqueeze(0), weights=weights)

    assert F.mse_loss(y_inr.squeeze(), y_tn.squeeze()) < EPS


@pytest.mark.parametrize(
    "n_models, num_samples, input_size, output_size, hidden_sizes, encoding_length",
    [
        (5, 32768, 1, 1, [16], 10),
        (4, 32768, 3, 2, [32, 4], 2),
        (2, 32768, 1, 1, [10, 20, 30], 16),
        (1, 32768, 2, 1, [128, 12], 1),
        (3, 32768, 1, 2, [1, 23, 12], 10),
        (7, 32768, 3, 1, [4, 6, 7, 8, 5, 7], 16),
        (6, 32768, 3, 2, [3, 3, 2, 1, 4, 2, 1, 4, 5, 3, 1, 2, 3, 4, 1, 2], 5),
    ],
)
def test_nerf_inr_tn_batch_equivalence(
    n_models: int,
    num_samples: int,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
    encoding_length: int,
) -> None:
    x_inr = [torch.rand((num_samples, input_size)) for _ in range(n_models)]
    x_tn = torch.stack(x_inr, dim=0)

    inrs = [
        NERF(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=True,
            mode=TargetNetworkMode.INR,
            encoding_length=encoding_length,
            learnable_encoding=False,
        )
        for _ in range(n_models)
    ]
    y_inr = torch.stack([model(x) for model, x in zip(inrs, x_inr)], dim=0)

    weights = {}
    for param_key in inrs[0].params.keys():
        if inrs[0].params[param_key].requires_grad:
            weights[param_key] = torch.stack([model.params[param_key] for model in inrs], dim=0)

    tn = NERF(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
        encoding_length=encoding_length,
        learnable_encoding=False,
    )
    y_tn = tn(x_tn, weights=weights)

    assert F.mse_loss(y_inr.squeeze(), y_tn.squeeze()) < EPS
