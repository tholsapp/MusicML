import numpy as np
from errno import ENOENT
from librosa import load
from os import strerror
from pathlib import Path

from librosa.feature.spectral import mfcc, spectral_bandwidth, \
        spectral_contrast, spectral_centroid, spectral_rolloff, \
        zero_crossing_rate

from music_ml_backend.exceptions.exceptions import AudioException


class Audio:
    def __init__(self, src):
        if self._is_valid_file(src):
            self.src = src

    def __str__(self):
        return f"<Audio src:{self.src!r}>"

    def _is_valid_file(self, src):
        if not Path(src).is_file():
            error_message = f""
            raise AudioException(
                    message=error_message,
                    errors=[FileNotFoundError(ENOENT, strerror(ENOENT), src)])
        return True

