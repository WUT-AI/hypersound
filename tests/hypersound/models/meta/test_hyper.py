import torch

from hypersound.cfg import TargetNetworkMode
from hypersound.models.meta.hyper import MLPHyperNetwork
from hypersound.models.nerf import NERF
from hypersound.models.siren import SIREN


def test_siren_hypernetwork() -> None:
    target_network = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[32, 16],
        mode=TargetNetworkMode.TARGET_NETWORK,
        bias=True,
        omega_0=30,
        omega_i=30,
        learnable_omega=False,
        gradient_fix=True,
    )
    hypernetwork = MLPHyperNetwork(
        target_network=target_network,
        shared_params=["o0", "o1", "o2"],
        input_size=32,
        layer_sizes=[64],
    )

    z = torch.rand((2, 32))
    x = torch.rand((2, 32, 1)) * 2 - 1

    weights = hypernetwork(z)
    y = target_network(x, weights=weights)

    assert y.shape == (2, 32, 1)


def test_nerf_hypernetwork() -> None:
    target_network = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[32, 16],
        mode=TargetNetworkMode.TARGET_NETWORK,
        bias=True,
        encoding_length=6,
        learnable_encoding=False,
    )
    hypernetwork = MLPHyperNetwork(
        target_network=target_network,
        shared_params=["freq"],
        input_size=32,
        layer_sizes=[64],
    )

    z = torch.rand((2, 32))
    x = torch.rand((2, 32, 1))

    weights = hypernetwork(z)
    y = target_network(x, weights=weights)

    assert y.shape == (2, 32, 1)


def test_siren_hypernetwork_with_learnable_omega() -> None:
    target_network = SIREN(
        input_size=1,
        output_size=1,
        hidden_sizes=[32, 16],
        bias=True,
        mode=TargetNetworkMode.TARGET_NETWORK,
        omega_0=30,
        omega_i=30,
        learnable_omega=True,
        gradient_fix=True,
    )
    hypernetwork = MLPHyperNetwork(
        target_network=target_network,
        shared_params=[],
        input_size=32,
        layer_sizes=[64],
    )

    z = torch.rand((2, 32))
    x = torch.rand((2, 32, 1)) * 2 - 1

    weights = hypernetwork(z)
    y = target_network(x, weights=weights)

    assert y.shape == (2, 32, 1)


def test_nerf_hypernetwork_with_learnable_encoding() -> None:
    target_network = NERF(
        input_size=1,
        output_size=1,
        hidden_sizes=[32, 16],
        mode=TargetNetworkMode.TARGET_NETWORK,
        bias=True,
        encoding_length=6,
        learnable_encoding=True,
    )
    hypernetwork = MLPHyperNetwork(
        target_network=target_network,
        shared_params=[],
        input_size=32,
        layer_sizes=[64],
    )

    z = torch.rand((2, 32))
    x = torch.rand((2, 32, 1)) * 2 - 1

    weights = hypernetwork(z)
    y = target_network(x, weights=weights)

    assert y.shape == (2, 32, 1)
