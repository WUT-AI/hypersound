from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from pytorch_yard.configs.cfg.lightning import LightningConf, LightningSettings


class Dataset(Enum):
    VCTK = auto()
    LJS = auto()
    GTZAN = auto()
    LIBRITTS = auto()
    LIBRISPEECH = auto()


class ModelType(Enum):
    NERF = auto()
    SIREN = auto()


class TargetNetworkMode(Enum):
    INR = auto()
    TARGET_NETWORK = auto()
    MODULATOR = auto()
    RESIDUAL = auto()


class LRScheduler(Enum):
    ONE_CYCLE = auto()
    CONSTANT = auto()


@dataclass
class DataSettings:
    num_workers: int = 8

    duration: float = 1.4861  # 1.4861 @ 22050 Hz --> 2^15 (32768) samples

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    data_norm: bool = True

    samples_per_epoch: int = 10_000

    file_limit: Optional[int] = None  # Limit on files used in training
    file_limit_validation: Optional[int] = 512

    # Boundaries for target network input indices. If provided, will rescale indices from [0-n_samples] to this value.
    # Setting value is normalized for a clip duration of 1.0 second if `proportional_index_scaling` is enabled.
    # A range of [-m, m] will use basal frequencies of up to m/π Hz, i.e. +/-300 ~ 100 Hz.
    index_scaling: Optional[tuple[float, float]] = (-1, 1)
    proportional_index_scaling: bool = False


@dataclass
class TransformSettings:
    start_offset: float = 0.5
    padding: bool = False
    dequantize: bool = True
    phase_mangle: bool = True
    random_crop: bool = True


@dataclass
class ModelSettings:
    type: ModelType = ModelType.NERF
    target_network_mode: TargetNetworkMode = TargetNetworkMode.TARGET_NETWORK

    layer_norm: bool = True
    encoder_channels: int = 16
    embedding_size: int = 32

    reconstruction_loss_lambda: float = 1.0

    activations_decay_loss_lambda: float = 0.0

    perceptual_loss_lambda: float = 1.0
    perceptual_loss_decay_epochs: Optional[int] = None
    perceptual_loss_final_lambda: float = 1.0
    perceptual_loss_freq_weights_p: float = 0.0
    perceptual_loss_freq_weights_warmup_epochs: int = 500

    fft_sizes: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128])
    hop_sizes: list[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    win_lengths: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128])

    hypernetwork_layer_sizes: list[int] = field(default_factory=lambda: [400, 768, 768, 768, 768, 400])

    target_network_shared_layers: Optional[list[int]] = None
    target_network_layer_sizes: list[int] = field(default_factory=lambda: 5 * [90])

    target_network_omega_0: float = 1.0
    target_network_omega_i: float = 1.0
    target_network_share_omega: bool = True
    target_network_learnable_omega: bool = False
    target_network_siren_gradient_fix: bool = True

    target_network_share_encoding: bool = True
    target_network_encoding_length: int = 10
    target_network_learnable_encoding: bool = False


@dataclass
class LogSettings:
    examples: int = 5  # Number of examples to plot to wandb during training/validation
    warmup_epochs: int = 100
    warmup_every_n_epoch: int = 1  # Controls how often logging is done in the initial training phase
    normal_every_n_epoch: int = 50  # Controls how often logging is done after warmup epochs have passed
    debug: bool = False


# Experiment settings validation schema & default values
@dataclass
class Settings(LightningSettings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    batch_size: int = 16
    scheduler: LRScheduler = LRScheduler.ONE_CYCLE
    learning_rate: float = 0.0001
    learning_rate_div_factor: float = 25.0  # or 25.0

    # ----------------------------------------------------------------------------------------------
    # Experiment logging settings
    # ----------------------------------------------------------------------------------------------
    log: LogSettings = field(default_factory=lambda: LogSettings())

    # ----------------------------------------------------------------------------------------------
    # Data settings
    # ----------------------------------------------------------------------------------------------
    dataset: Dataset = Dataset.VCTK
    data: DataSettings = field(default_factory=lambda: DataSettings())
    transforms: TransformSettings = field(default_factory=lambda: TransformSettings())

    # ----------------------------------------------------------------------------------------------
    # Model settings
    # ----------------------------------------------------------------------------------------------
    model: ModelSettings = field(default_factory=lambda: ModelSettings())

    # ----------------------------------------------------------------------------------------------
    # PyTorch Lightning overrides
    # ----------------------------------------------------------------------------------------------
    pl: LightningConf = LightningConf(
        max_epochs=2500,
        check_val_every_n_epoch=100,
        deterministic=False,
    )

    validate_before_training: bool = False
