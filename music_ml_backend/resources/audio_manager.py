from errno import ENOENT
from glob import glob
from os import strerror
from pathlib import Path

from music_ml_backend.exceptions.exceptions import AudioException

class AudioManager:

    @classmethod
    def get(cls, fn, fdir):
        try:
            return Audio(fdir + fn)
        except FileNotFoundError as e:
            error_message = f"file {fn!r} not found in directory {fdir!r}"
            raise AudioExcpetion(message=error_message, errors=[e])

    @classmethod
    def get_all(cls, fdir):
        audio_files = []
        try:
            return [Audio(fls) for fls in audio_files]
        except FileNotFoundError:
            raise

    @classmethod
    def save(cls, f, fdir):
        raise NotImplementedError

    @classmethod
    def batch_get(cls, fdir, sz):
        raise NotImplementedError

    @classmethod
    def batch_save(cls, fls, fdir):
        raise NotImplementedError

