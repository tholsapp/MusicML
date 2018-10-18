import logging
import os

from errno import ENOENT
from glob import glob
from os import strerror
from pathlib import Path

from config import MusicMLConfig
from music_ml_backend.exceptions.exceptions import AudioException
from music_ml_backend.resources.audio import Audio

log = logging.getLogger(__name__)

class AudioManager:

    @classmethod
    def get(cls, fn, fdir):
        try:
            log.info(f"Accessing file {fn!r}")
            return Audio(fdir + "/" + fn)
        except FileNotFoundError as e:
            error_message = f"file {fn!r} not found in directory {fdir!r}"
            raise AudioExcpetion(message=error_message, errors=[e])

    @classmethod
    def get_genre_map(cls, genres_src):
        genre_map = {}

        log.info("Initializing dictionary: mapping genre to audio files")
        for genre_name in MusicMLConfig.GENRE_NAMES:
            current_src = genres_src + "/" + genre_name
            os.chdir(current_src)
            genre_map[genre_name] = []
            for f in os.listdir():
                genre_map[genre_name].append(AudioManager.get(f, current_src))

        return genre_map


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

