import pytest
import torch
import torch.nn
import torch.nn.functional as F

from hypersound.cfg import TargetNetworkMode
from hypersound.models.siren import SIREN

EPS = 1e-3


@pytest.mark.parametrize(
    "num_samples",
    [
        1,
        50,
        32768,
    ],
)
def test_siren_standard_inference(num_samples: int) -> None:
    model = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.INR,
        omega_0=1.0,
        omega_i=1.0,
        learnable_omega=False,
        gradient_fix=True,
    )
    x = torch.rand((num_samples, 1))

    assert model(x, weights=None).shape == (num_samples, 1)


@pytest.mark.parametrize(
    "num_samples",
    [
        1,
        50,
        32768,
    ],
)
def test_siren_train_loop(num_samples: int) -> None:
    model = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        mode=TargetNetworkMode.INR,
        bias=True,
        omega_0=30.0,
        omega_i=1.0,
        learnable_omega=False,
        gradient_fix=True,
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.L1Loss()
    x = torch.ones((num_samples, 1))
    y = torch.rand((num_samples, 1))
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
def test_siren_inr_tn_equivalence(num_samples: int) -> None:
    x = torch.rand((num_samples, 1))

    inr = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.INR,
        omega_0=30.0,
        omega_i=1,
        learnable_omega=False,
        gradient_fix=True,
    )
    y_inr = inr(x)

    tn = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[16],
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
        omega_0=30.0,
        omega_i=1,
        learnable_omega=False,
        gradient_fix=True,
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
def test_siren_inr_tn_batch_equivalence(
    n_models: int,
    num_samples: int,
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
) -> None:
    x_inr = [torch.rand((num_samples, input_size)) for _ in range(n_models)]
    x_tn = torch.stack(x_inr, 0)

    inrs = [
        SIREN(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            bias=True,
            mode=TargetNetworkMode.INR,
            omega_0=30.0,
            omega_i=30.0,
            learnable_omega=False,
            gradient_fix=True,
        )
        for _ in range(n_models)
    ]
    y_inr = torch.stack([model(x) for model, x in zip(inrs, x_inr)], dim=0)

    weights = {}
    for param_key in inrs[0].params.keys():
        if inrs[0].params[param_key].requires_grad:
            weights[param_key] = torch.stack([model.params[param_key] for model in inrs], dim=0)

    tn = SIREN(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
        omega_0=30.0,
        omega_i=30.0,
        learnable_omega=False,
        gradient_fix=True,
    )
    y_tn = tn(x_tn, weights=weights)

    assert F.mse_loss(y_inr.squeeze(), y_tn.squeeze()) < EPS
