from pathlib import Path
from typing import Any

from torchaudio.datasets.gtzan import GTZAN

from hypersound.datasets.base import BaseSamples

ORIGINAL_AUDIO_EXT = ".wav"
VALIDATION_SPLIT = 0.05


class GTZAN_Samples(BaseSamples):
    def __init__(
        self,
        root: str,
        download: bool,
        sample_rate: int,
        fold: str,
        **kwargs: Any,
    ):

        super().__init__(sample_rate, fold, **kwargs)

        GTZAN(root, download=download)
        src_dir = Path(root) / "genres"
        dst_dir = Path(root) / "genres"

        self.process_recordings_dir(src_dir, dst_dir, ORIGINAL_AUDIO_EXT, validation_split=VALIDATION_SPLIT)
