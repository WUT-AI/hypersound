import pytest
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor

from hypersound.cfg import TargetNetworkMode
from hypersound.models.meta.inr import INR

EPS = 1e-3


@pytest.mark.parametrize(
    "num_samples",
    [
        1,
        50,
        32768,
    ],
)
def test_inr_standard_inference(num_samples: int) -> None:
    model = INR(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        activation_fn=torch.nn.ReLU(),
        bias=True,
        mode=TargetNetworkMode.INR,
    )

    # x --> (num_samples, 1)
    x: Tensor = torch.rand((num_samples, 1))

    assert model(x, weights=None).shape == (num_samples, 1)


@pytest.mark.parametrize(
    "num_samples",
    [
        1,
        50,
        32768,
    ],
)
def test_inr_train_loop(num_samples: int) -> None:
    model = INR(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        activation_fn=torch.nn.ReLU(),
        bias=True,
        mode=TargetNetworkMode.INR,
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.L1Loss()
    x = torch.rand((num_samples, 1))
    y = torch.ones((num_samples, 1))
    y0 = model(x)

    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    assert F.mse_loss(y.squeeze(), y0.squeeze()) > EPS


@pytest.mark.parametrize(
    "num_samples",
    [
        1,
        50,
        32768,
    ],
)
def test_inr_tn_equivalence(num_samples: int) -> None:
    x = torch.rand((num_samples, 1))

    inr = INR(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        activation_fn=torch.nn.ReLU(),
        bias=True,
        mode=TargetNetworkMode.INR,
    )
    y_inr = inr(x)

    tn = INR(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        activation_fn=torch.nn.ReLU(),
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
    )
    weights = {name: param.unsqueeze(0) for name, param in inr.params.items()}
    y_tn = tn(x.unsqueeze(0), weights=weights)

    assert F.mse_loss(y_inr.squeeze(), y_tn.squeeze()) < EPS


@pytest.mark.parametrize(
    "n_models, num_samples, input_size, output_size, hidden_sizes",
    [
        (5, 32768, 1, 1, [16]),
        (4, 32768, 3, 2, [32, 4]),
        (2, 32768, 1, 1, [10, 20, 30]),
        (1, 32768, 2, 1, [128, 12]),
        (3, 32768, 1, 2, [1, 23, 12]),
        (7, 32768, 3, 1, [4, 6, 7, 8, 5, 7]),
        (6, 32768, 3, 2, [3, 3, 2, 1, 4, 2, 1, 4, 5, 3, 1, 2, 3, 4, 1, 2]),
    ],
)
def test_inr_tn_batch_equivalence(
    n_models: int,
    num_samples: int,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
) -> None:
    x_inr = [torch.rand((num_samples, input_size)) for _ in range(n_models)]
    x_tn = torch.stack(x_inr, 0)

    inrs = [
        INR(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation_fn=torch.nn.ReLU(),
            bias=True,
            mode=TargetNetworkMode.INR,
        )
        for _ in range(n_models)
    ]
    y_inr = torch.stack([inr(x) for inr, x in zip(inrs, x_inr)], dim=0)

    weights = {}
    for param_key in inrs[0].params.keys():
        weights[param_key] = torch.stack([model.params[param_key] for model in inrs], dim=0)

    tn = INR(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        activation_fn=torch.nn.ReLU(),
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
    )
    y_tn = tn(x_tn, weights=weights)

    assert F.mse_loss(y_inr.squeeze(), y_tn.squeeze()) < EPS
