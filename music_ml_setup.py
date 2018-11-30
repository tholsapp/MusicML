import glob
import logging
import os
import random
import sys

from config import MusicMLConfig

log = logging.getLogger(__name__)


class MusicMLSetup():

    def create_project_structure():
        """
        Creates directory structure for data
        """
        if os.path.isdir(MusicMLConfig.TRAINING_SRC) and \
           os.path.isdir(MusicMLConfig.VALIDATION_SRC) and \
           os.path.isdir(MusicMLConfig.TESTING_SRC):
            log.info("Project Structure is already setup.")
        else:
            log.info("Setting up Project Structure...")
            # create training directory
            log.info(f"Created directory : {MusicMLConfig.TRAINING_SRC}")
            os.mkdir(MusicMLConfig.TRAINING_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                os.mkdir(MusicMLConfig.TRAINING_SRC + "/" + genre)
            # create validation directory
            log.info(f"Created directory : {MusicMLConfig.VALIDATION_SRC}")
            os.mkdir(MusicMLConfig.VALIDATION_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                os.mkdir(MusicMLConfig.VALIDATION_SRC + "/" + genre)
            # create testing directory
            log.info(f"Created directory : {MusicMLConfig.TESTING_SRC}")
            os.mkdir(MusicMLConfig.TESTING_SRC)
            for genre in MusicMLConfig.GENRE_LABELS:
                os.mkdir(MusicMLConfig.TESTING_SRC + "/" + genre)

    def convert_and_split_data():
        """
        Converts data from au to wav format and splits data into
        training (60%), validation (20%), and testing(20%)
        """
        os.chdir(MusicMLConfig.RAW_DATA_SRC) # change to raw data directory

        genres_to_raw = {label : [] for label in MusicMLConfig.GENRE_LABELS}

        # get all raw data
        for genre_label in os.listdir(MusicMLConfig.RAW_DATA_SRC):
            if os.path.isdir(genre_label):
                for f in os.listdir(genre_label):
                    genres_to_raw[genre_label].append(str(f))

            # remove everything in formatted directory
        for label in MusicMLConfig.GENRE_LABELS:
            training_files = glob.glob(f"{MusicMLConfig.TRAINING_SRC}/{label}/*")
            for f in training_files:
                os.remove(f)
            validation_files = glob.glob(f"{MusicMLConfig.VALIDATION_SRC}/{label}/*")
            for f in validation_files:
                os.remove(f)
            testing_files = glob.glob(f"{MusicMLConfig.TESTING_SRC}/{label}/*")
            for f in testing_files:
                os.remove(f)


        # create list of commands to run
        sox_cmds = []
        # convert and split data
        for label in genres_to_raw:
            audio_src = genres_to_raw[label]
            # randomly shuffle audio srcs
            random.shuffle(audio_src)

            # training dataset
            #for i in range(0, 60):
            for i in range(0, 70):
                orig_file = str(audio_src[i])
                new_file = str(audio_src[i][:-3]) + ".wav"
                new_file_src = MusicMLConfig.TRAINING_SRC + '/' + \
                            label + '/' + new_file
                sox_cmd = label + '/' + orig_file + " " + new_file_src
                sox_cmds.append(sox_cmd)

            # validation dataset
            #for i in range(60, 80):
            #    orig_file = str(audio_src[i])
            #    new_file = str(audio_src[i][:-3]) + ".wav"
            #    new_file_src = MusicMLConfig.VALIDATION_SRC + '/' + \
                    #                label + '/' + new_file
            #    sox_cmd = label + '/' + orig_file + " " + new_file_src
            #    sox_cmds.append(sox_cmd)


            # testing dataset
            for i in range(70, 100):
                orig_file = str(audio_src[i])
                new_file = str(audio_src[i][:-3]) + ".wav"
                new_file_src = MusicMLConfig.TESTING_SRC + '/' + \
                            label + '/' + new_file
                sox_cmd = label + '/' + orig_file + " " + new_file_src
                sox_cmds.append(sox_cmd)


        for sox_cmd in sox_cmds:
            os.system(f" sox {sox_cmd}")
            log.info(f"Running : sox {sox_cmd}")

