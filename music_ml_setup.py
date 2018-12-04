import glob
import logging
import os
import random
import sys
import zipfile

from config import MusicMLConfig

log = logging.getLogger(__name__)


class MusicMLSetup():

    def create_project_structure():
        """
        Creates directory structure for data
        """
        if os.path.isdir(MusicMLConfig.TRAINING_SRC) and \
           os.path.isdir(MusicMLConfig.WAV_FORMAT_SRC) and \
           os.path.isdir(MusicMLConfig.TESTING_SRC):
            log.info("Project is already setup.\n" +
                    "Delete /wav_format /testing / training to recreate project structure")
        else:
            log.info("Setting up Project Structure...")
            # create upload directory
            log.info(f"Created directory : {MusicMLConfig.UPLOAD_SRC}")
            os.mkdir(MusicMLConfig.UPLOAD_SRC)
            # create au format directory
            log.info(f"Created directory : {MusicMLConfig.AU_SRC}")
            os.mkdir(MusicMLConfig.AU_SRC)
            # create training directory
            log.info(f"Created directory : {MusicMLConfig.TRAINING_SRC}")
            os.mkdir(MusicMLConfig.TRAINING_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                log.info(f"Created directory : {MusicMLConfig.TRAINING_SRC}/{genre}")
                os.mkdir(MusicMLConfig.TRAINING_SRC + "/" + genre)
            # create validation directory
            log.info(f"Created directory : {MusicMLConfig.WAV_FORMAT_SRC}")
            os.mkdir(MusicMLConfig.WAV_FORMAT_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                log.info(f"Created directory : {MusicMLConfig.WAV_FORMAT_SRC}/{genre}")
                os.mkdir(MusicMLConfig.WAV_FORMAT_SRC + "/" + genre)
            # create testing directory
            log.info(f"Created directory : {MusicMLConfig.TESTING_SRC}")
            os.mkdir(MusicMLConfig.TESTING_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                log.info(f"Created directory : {MusicMLConfig.TESTING_SRC}/{genre}")
                os.mkdir(MusicMLConfig.TESTING_SRC + "/" + genre)
            # convert raw data
            # and seperate data into training, validation, and testing
            MusicMLSetup._unzip_audio_zip()
            MusicMLSetup._convert_and_split_data()
            MusicMLSetup.shuffle_training_testing()

    def _unzip_audio_zip():
        log.info("Unzipping audio files")
        # unzip au format audo files
        with zipfile.ZipFile(MusicMLConfig.ZIP_SRC, 'r') as zfile:
            zfile.extractall(MusicMLConfig.AU_FORMAT_SRC[:-7])

    def _convert_and_split_data():
        """
        Converts data from au to wav format and splits data into
        training (60%), validation (20%), and testing(20%)
        """

        os.chdir(MusicMLConfig.AU_FORMAT_SRC) # change to raw data directory

        genres_to_raw = {label : [] for label in MusicMLConfig.GENRE_LABELS}

        # get all raw data
        for genre_label in os.listdir(MusicMLConfig.AU_FORMAT_SRC):
            if os.path.isdir(genre_label):
                for f in os.listdir(genre_label):
                    genres_to_raw[genre_label].append(str(f))

        # create list of sox commands
        sox_cmds = []
        # convert and split data
        for label in genres_to_raw:
            audio_src = genres_to_raw[label]
            # randomly shuffle audio srcs
            random.shuffle(audio_src)

            for src in audio_src:
               orig_file = str(src)
               new_file = str(src[:-3]) + ".wav"
               new_file_src = MusicMLConfig.WAV_FORMAT_SRC + '/' + \
                           label + '/' + new_file
               sox_cmd = label + '/' + orig_file + " " + new_file_src
               sox_cmds.append(sox_cmd)

        # execute sox commands
        for sox_cmd in sox_cmds:
            os.system(f" sox {sox_cmd}")
            log.info(f"Running : sox {sox_cmd}")

    def shuffle_training_testing():
        # remove everything in formatted directory
        for label in MusicMLConfig.GENRE_LABELS:
            training_files = glob.glob(f"{MusicMLConfig.TRAINING_SRC}/{label}/*")
            for f in training_files:
                log.info(f" Removing {f}")
                os.remove(f)
            testing_files = glob.glob(f"{MusicMLConfig.TESTING_SRC}/{label}/*")
            for f in testing_files:
                log.info(f" Removing {f}")
                os.remove(f)

        for label in os.listdir(MusicMLConfig.WAV_FORMAT_SRC):
            cmds = []
            path = f"{MusicMLConfig.WAV_FORMAT_SRC}/{label}"
            if os.path.isdir(path):
                log.info(f"Splitting Testing and Training Data : {path}")
                files = []
                for f in os.listdir(path):
                    files.append(f"{f}")
                random.shuffle(files)
                for i in range(0, 70):
                    cmds.append(f"cp {path}/{files[i]} {MusicMLConfig.TRAINING_SRC}/{label}/{files[i]}")
                for i in range(70, 100):
                    cmds.append(f"cp {path}/{files[i]} {MusicMLConfig.TESTING_SRC}/{label}/{files[i]}")
                for cmd in cmds:
                    os.system(f"{cmd}")
                    log.info(f"Running : {cmd}")

