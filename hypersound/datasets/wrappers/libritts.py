from pathlib import Path
from typing import Any

from torchaudio.datasets.libritts import LIBRITTS

from hypersound.datasets.base import BaseSamples

ORIGINAL_AUDIO_EXT = ".wav"
VAL_SPEAKER_SPLIT = 10


class LibriTTS_Samples(BaseSamples):
    def __init__(
        self,
        root: str,
        download: bool,
        sample_rate: int,
        fold: str,
        **kwargs: Any,
    ):

        super().__init__(sample_rate, fold, **kwargs)

        LIBRITTS(root, download=download, url="train-clean-360")
        src_dir = Path(root) / "LibriTTS/train-clean-360"
        dst_dir = Path(root) / "LibriTTS-train-clean-360"

        self.process_recordings_dir(src_dir, dst_dir, ORIGINAL_AUDIO_EXT, val_speaker_split=VAL_SPEAKER_SPLIT)
