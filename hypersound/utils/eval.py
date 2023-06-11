import copy
from math import ceil
from pathlib import Path
from typing import Optional, cast

import soundfile as sf
import torch
from more_itertools import chunked
from omegaconf import OmegaConf
from pathos.threading import ThreadPool as Pool
from pytorch_yard import RootConfig
from rich import print
from rich.progress import track
from torch import Tensor

from hypersound.cfg import Settings
from hypersound.datasets.audio import AudioDataset
from hypersound.datasets.utils import init_datamodule
from hypersound.systems.main import HyperNetworkAE


def load_directory(recording_dir: Path, audio_extension: str) -> list[Path]:
    recordings = list(recording_dir.rglob(f"*.{audio_extension}"))
    recordings = sorted(recordings, key=lambda path: str(path.name))
    return recordings


def load_recordings(reference_dir: Path, generated_dir: Path, audio_extension: str) -> tuple[list[Path], list[Path]]:
    reference = load_directory(reference_dir, audio_extension)
    generated = load_directory(generated_dir, audio_extension)

    assert len(reference) == len(generated), "Number of reference and generated recordings differs."
    assert len(reference) > 0, "Data directory is empty."

    return reference, generated


class EvalModel:
    def __init__(
        self,
        cfg_path: Path,
        ckpt_path: Path,
        audio_format: str,
        is_cuda: bool,
        batch_size: int,
        sample_rate: Optional[int] = None,
        file_limit_override: Optional[int] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.is_cuda = is_cuda
        self.batch_size = batch_size
        self.file_limit_override = file_limit_override

        self.cfg: Settings

        self._load_config()
        self._setup()
        self._load_checkpoint()

    def _load_config(self):
        # Config
        try:
            self.root_cfg = cast(RootConfig, OmegaConf.load(self.cfg_path))
        except Exception:
            raise RuntimeError(f"{self.cfg_path} is not a valid config file.")

        self.cfg = cast(Settings, OmegaConf.merge(OmegaConf.structured(Settings), self.root_cfg.cfg))

        if self.file_limit_override is not None:
            self.cfg.data.file_limit_validation = self.file_limit_override or None

        print(f"Loaded config: {OmegaConf.to_yaml(self.root_cfg, resolve=True)}")

    def _setup(self):
        # Model setup
        self.datamodule, self.interpolation_dm = init_datamodule(self.root_cfg, self.sample_rate)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.system = HyperNetworkAE(
            cfg=self.cfg,
            input_length=self.datamodule.train.shape[1][0],
            spec_transform=copy.deepcopy(self.datamodule.train.spec_transform),
        )

    def _load_checkpoint(self):
        # Checkpoint
        checkpoint = self.ckpt_path

        if checkpoint.suffix != ".ckpt":
            print(f"Searching for a valid checkpoint file in: {self.ckpt_path}")
            if not self.ckpt_path.is_dir():
                raise RuntimeError(f"{self.ckpt_path} is not a directory.")

            checkpoint = [name for name in self.ckpt_path.iterdir() if name.suffix == ".ckpt"][-1]

        if not checkpoint.is_file():
            raise RuntimeError(f"{checkpoint} is not a valid file.")

        print(f"Loading checkpoint {checkpoint}...")
        self.system = HyperNetworkAE.load_from_checkpoint(str(checkpoint))  # type: ignore

        if self.is_cuda:
            self.system.to("cuda")

        print(self.system)

    def generate_recordings(self, reference_dir: Path, generated_dir: Path, generate_training: bool = False) -> None:
        print("Verifying recording files...")

        if generate_training:
            self._generate_recordings("train", reference_dir, generated_dir)
        self._generate_recordings("validation", reference_dir, generated_dir)

    def _generate_recordings(self, fold: str, reference_dir: Path, generated_dir: Path) -> None:
        dataset: AudioDataset = getattr(self.datamodule, fold)
        reference_dir = reference_dir / fold
        generated_dir = generated_dir / fold

        if reference_dir.is_dir():
            assert len(list(reference_dir.iterdir())) == len(dataset), \
                "The reference directory and source dataset have a different number of recordings."  # fmt: skip
        else:
            self._process_dataset(dataset, fold, reference_dir, generate=False)

        if generated_dir.is_dir():
            assert len(list(generated_dir.iterdir())) == len(dataset), \
                "The generated directory has an invalid number of recordings."  # fmt: skip
        else:
            self._process_dataset(dataset, fold, generated_dir, generate=True)

    def _save_recording(self, output_dir: Path, idx: int, signal: Tensor):
        sf.write(  # type: ignore
            str(output_dir / f"{idx}.{self.audio_format}"),
            signal.detach().cpu().numpy(),
            self.sample_rate or self.cfg.data.sample_rate,
            format=self.audio_format,
            subtype="PCM_24",
        )

    def _process_dataset(self, dataset: AudioDataset, fold: str, output_dir: Path, generate: bool):
        output_dir.mkdir(parents=True)

        description = "Generated" if generate else "Reference"

        p = Pool(self.batch_size)

        for idxs in track(
            chunked(range(len(dataset)), self.batch_size),
            description=f"Recreating: {description} / {fold}",
            total=ceil(len(dataset) / self.batch_size),
        ):

            indices = torch.stack([dataset[idx][0] for idx in idxs])
            signal = torch.stack([dataset[idx][1] for idx in idxs])

            assert self.interpolation_dm

            if generate:
                if self.sample_rate and fold == "validation":
                    # evaluate on interpolated indices, base input signal remains the same
                    indices = torch.stack([self.interpolation_dm.validation[idx][0] for idx in idxs])

                if self.is_cuda:
                    indices = indices.to("cuda")
                    signal = signal.to("cuda")

                signal = self.system.reconstruct(indices, signal)
            elif self.sample_rate and fold == "validation":  # ground truth for interpolation
                signal = torch.stack([self.interpolation_dm.validation[idx][1] for idx in idxs])

            p.map(  # type: ignore
                self._save_recording,
                [output_dir] * self.batch_size,
                idxs,
                signal.unbind(0),
            )
