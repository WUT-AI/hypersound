from functools import partial
from typing import Any, Callable, Optional, cast

import torch.utils.data
from omegaconf import OmegaConf
from pytorch_yard import RootConfig
from torch import Tensor

from hypersound.cfg import Dataset, Settings
from hypersound.datamodules.audio import AudioDataModule
from hypersound.datasets.wrappers.gtzan import GTZAN_Samples
from hypersound.datasets.wrappers.librispeech import LibriSpeech_Samples
from hypersound.datasets.wrappers.libritts import LibriTTS_Samples
from hypersound.datasets.wrappers.ljs import LJS_Samples
from hypersound.datasets.wrappers.vctk import VCTK_Samples


def init_datamodule(
    root_cfg: RootConfig, resample_rate: Optional[int] = None
) -> tuple[AudioDataModule, Optional[AudioDataModule]]:
    cfg = cast(Settings, root_cfg.cfg)
    kwargs = dict(
        root=root_cfg.data_dir,
        download=True,
        duration=cfg.data.duration,
        padding=cfg.transforms.padding,
        dequantize=cfg.transforms.dequantize,
        phase_mangle=cfg.transforms.phase_mangle,
        random_crop=cfg.transforms.random_crop,
        start_offset=cfg.transforms.start_offset,
    )

    _dataset: Callable[[str], torch.utils.data.Dataset[tuple[Tensor, int]]]
    if cfg.dataset is Dataset.VCTK:
        _dataset = partial(VCTK_Samples, **kwargs)
    elif cfg.dataset is Dataset.LJS:
        _dataset = partial(LJS_Samples, **kwargs)
    elif cfg.dataset is Dataset.GTZAN:
        _dataset = partial(GTZAN_Samples, **kwargs)
    elif cfg.dataset is Dataset.LIBRITTS:
        _dataset = partial(LibriTTS_Samples, **kwargs)
    elif cfg.dataset is Dataset.LIBRISPEECH:
        _dataset = partial(LibriSpeech_Samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}.")

    train_dataset = _dataset(fold="train", sample_rate=cfg.data.sample_rate)  # type: ignore
    validation_dataset = _dataset(fold="validation", sample_rate=cfg.data.sample_rate)  # type: ignore

    main_dm = AudioDataModule(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=cfg.batch_size,
        **cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True)),
    )

    if resample_rate:
        # Prepare datamodule for resampling evaluation
        _cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
        _cfg.update({"sample_rate": resample_rate})
        interpolation_dm = AudioDataModule(
            train_dataset=train_dataset,
            validation_dataset=_dataset(fold="validation", sample_rate=resample_rate),  # type: ignore
            batch_size=cfg.batch_size,
            **_cfg,
        )
        interpolation_dm.prepare_data()
        interpolation_dm.setup()
    else:
        interpolation_dm = None

    return main_dm, interpolation_dm
