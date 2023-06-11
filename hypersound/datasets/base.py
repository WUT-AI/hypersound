import random
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Optional, cast

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torchaudio
from more_itertools import chunked
from pathos.threading import ThreadPool as Pool
from pytorch_yard.utils.logging import info_bold
from rich.progress import Progress
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from hypersound.datasets.transforms import (
    Dequantize,
    RandomApply,
    RandomCrop,
    RandomPhaseMangle,
    Transform,
)

PROCESSING_BATCH_SIZE = 8


class BaseSamples(Dataset[tuple[Tensor, int]], ABC):
    def __init__(
        self,
        sample_rate: int,
        fold: str,
        duration: float,
        start_offset: float,
        padding: bool,
        dequantize: bool,
        phase_mangle: bool,
        random_crop: bool,
        transforms: Optional[transforms.Compose] = None,
    ):

        self._recordings: list[Path]

        assert 8000 <= sample_rate <= 48000, f"Sample rate {sample_rate} is out of expected range of 8-48 kHz"

        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(duration * sample_rate)
        self.start_offset = int(start_offset * sample_rate)
        self.padding = padding
        self.dequantize = dequantize
        self.phase_mangle = phase_mangle
        self.random_crop = random_crop

        info_bold(f"Sample rate: {sample_rate}, duration: {duration:.2f} --> N_SIGNAL = {self.total_samples}")

        assert fold in ["train", "validation"]
        self.fold = fold

        self.transforms = transforms or self.default_transforms()

        self.suffixes = f"-{sample_rate}-{self.total_samples}"
        self.suffixes = self.suffixes + ("p" if self.padding else "")
        self.suffixes = self.suffixes + (f"s{int(self.start_offset)}" if self.start_offset else "")

    @staticmethod
    def _save_recording(
        recording_src: Path,
        audio_dst_dir: Path,
        sample_rate: int,
        segment_len: int,
        start_offset: int,
        padding: bool,
    ):
        signal, _ = librosa.load(str(recording_src), sr=sample_rate)  # type: ignore
        signal = cast(npt.NDArray[np.float32], signal)

        signal = signal[start_offset:]

        if padding:
            pad = segment_len - len(signal) % segment_len
            signal = cast(npt.NDArray[np.float32], np.pad(signal, (0, pad)))  # type: ignore
        else:
            signal = signal[: len(signal) - len(signal) % segment_len]

        signal = signal.reshape(-1, segment_len)

        for i, y in enumerate(signal):
            dst_name = recording_src.with_stem(f"{recording_src.stem}-{i + 1}").with_suffix(".wav").name
            sf.write(  # type: ignore
                str(audio_dst_dir / dst_name),
                y,
                sample_rate,
                format="wav",
                subtype="PCM_16",
            )

    def __len__(self):
        return len(self._recordings)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        signal, sample_rate = cast(tuple[Tensor, int], torchaudio.load(str(self._recordings[idx])))  # type: ignore
        if self.transforms is not None:
            signal = cast(Tensor, self.transforms(signal))
        return signal, sample_rate

    def process_recordings_dir(
        self,
        src_dir: Path,
        dst_dir: Path,
        original_audio_ext: str,
        validation_split: Optional[float] = None,
        val_speaker_split: Optional[int] = None,
    ):
        dst_dir = Path(str(dst_dir) + self.suffixes)

        if not dst_dir.is_dir():
            info_bold("Preparing processed version of the dataset...")
            p = Pool(PROCESSING_BATCH_SIZE)

            dst_dir.mkdir()

            with Progress() as progress:

                def process_recordings(recordings: list[Path], fold: str):
                    (dst_dir / fold).mkdir(exist_ok=True)

                    progress.update(
                        progress_recording,
                        total=len(recordings),
                        completed=0,
                        description=f"[red]Processing {fold} recordings...",
                    )

                    _save = partial(
                        self._save_recording,
                        audio_dst_dir=dst_dir / fold,
                        sample_rate=self.sample_rate,
                        segment_len=self.total_samples * 2,
                        start_offset=self.start_offset,
                        padding=self.padding,
                    )

                    for recording_batch in chunked(recordings, PROCESSING_BATCH_SIZE):
                        p.map(_save, recording_batch)  # type: ignore
                        progress.update(progress_recording, advance=len(recording_batch))

                if validation_split:
                    recordings = sorted([path for path in src_dir.glob(f"*{original_audio_ext}") if not path.is_dir()])
                    random.Random(1).shuffle(recordings)

                    progress_recording = progress.add_task("[red]Processing recordings...", total=len(recordings))

                    val_size = int(validation_split * len(recordings))
                    recordings_train = recordings[:-val_size]
                    recordings_val = recordings[-val_size:]

                    process_recordings(recordings_train, "train")
                    process_recordings(recordings_val, "validation")
                elif val_speaker_split:
                    speakers = sorted([path for path in src_dir.iterdir() if path.is_dir()])

                    progress_speaker = progress.add_task("[red]Processing speakers...", total=len(speakers))
                    progress_recording = progress.add_task("[yellow]Resampling speaker recordings...", total=0)

                    speakers_train = speakers[:-val_speaker_split]
                    speakers_val = speakers[-val_speaker_split:]

                    for speaker in speakers_train:
                        process_recordings(list(speaker.rglob(f"*{original_audio_ext}")), "train")
                        progress.update(progress_speaker, advance=1)
                    for speaker in speakers_val:
                        process_recordings(list(speaker.rglob(f"*{original_audio_ext}")), "validation")
                        progress.update(progress_speaker, advance=1)
                else:
                    raise RuntimeError("Both `validation_split` and `val_speaker_split` are unspecified.")

            p.close()  # type: ignore

        files = list((dst_dir / self.fold).glob("*.wav"))
        files.sort()
        self._recordings = files

    def default_transforms(self):
        _transforms: list[Transform] = []

        if self.fold == "validation":
            _transforms.append(RandomCrop(self.total_samples, random=False))
        else:
            _transforms.append(RandomCrop(self.total_samples, random=self.random_crop))
            if self.phase_mangle:
                _transforms.append(
                    RandomApply(
                        RandomPhaseMangle(
                            min_f=20,
                            max_f=2000,
                            amplitude=0.99,
                            sample_rate=self.sample_rate,
                        ),
                        p=0.8,
                    )
                )

            if self.dequantize:
                _transforms.append(Dequantize(16))

        return transforms.Compose(_transforms)
