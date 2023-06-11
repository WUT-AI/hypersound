from pathlib import Path
from typing import Any

from torchaudio.datasets.vctk import VCTK_092

from hypersound.datasets.base import BaseSamples

ORIGINAL_AUDIO_EXT = ".flac"
VAL_SPEAKER_SPLIT = 10


class VCTK_Samples(BaseSamples):
    def __init__(
        self,
        root: str,
        download: bool,
        sample_rate: int,
        fold: str,
        **kwargs: Any,
    ):

        super().__init__(sample_rate, fold, **kwargs)

        VCTK_092(root, download=download)
        src_dir = Path(root) / "VCTK-Corpus-0.92/wav48_silence_trimmed"
        dst_dir = Path(root) / "VCTK-Corpus-0.92"

        self.process_recordings_dir(src_dir, dst_dir, ORIGINAL_AUDIO_EXT, val_speaker_split=VAL_SPEAKER_SPLIT)
