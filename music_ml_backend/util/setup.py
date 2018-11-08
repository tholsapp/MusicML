import logging
import os
import sys

from config import MusicMLConfig

log = logging.getLogger(__name__)


def convert_au_to_wav():
    os.chdir(MusicMLConfig.RAW_DATA_SRC)  # change to raw data directory

    log.info("Initializing Data Conversion")
    # for each genre with au audio format
    for genre_dir in os.listdir(MusicMLConfig.RAW_DATA_SRC):
        print(genre_dir)
        if(os.path.isdir(genre_dir)):
            # each au audio file within this directory
            new_directory = MusicMLConfig.FORMATTED_DATA_SRC + '/' + genre_dir
            if not os.path.isdir(new_directory):
                log.info(f"Creating directory {new_directory!r}")
                os.system(f"mkdir {new_directory!r}")
            else:
                log.info(f"Directory {new_directory!r} already exists")
            for f in os.listdir(genre_dir):
                orig_file = str(f)
                new_file = str(f[:-3]) + ".wav"
                new_file_src = MusicMLConfig.FORMATTED_DATA_SRC + '/' + \
                        genre_dir + '/' + new_file

                if not os.path.exists(new_file_src):
                    sox_cmd = genre_dir + '/' + orig_file + " " + new_file_src

                    # convert audio file and save
                    log.info(f"Converting {orig_file!r} to {new_file!r}")
                    os.system(f"sox {sox_cmd}")
                else:
                    log.info(f"{new_file!r} already exists")

