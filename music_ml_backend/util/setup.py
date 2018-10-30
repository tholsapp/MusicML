import os
import sys

from config import MusicMLConfig


def convert_au_to_wav(genre_dirs):
    os.chdir(genre_dirs)  # change to this directory

    i = 0

    # for each genre with au audio format
    for genre_dir in os.listdir(genre_dirs):
        if(os.path.isdir(genre_dir)):
            # each au audio file within this directory
            for f in os.listdir(genre_dir):
                if i == 10:
                    i = 0

                # split trianing (70%) and testing (30%)
                if i < 7:
                    os.system("sox " + genre_dir + '/' + str(f) + " " +
                            MusicMLConfig.TRAIN_GENRES_SRC + '/' + genre_dir + '/' + str(f[:-3]) + ".wav")
                else:
                     os.system("sox " + genre_dir + '/' + str(f) + " " +
                            MusicMLConfig.TEST_DATA_DIR + '/' + genre_dir + '/' + str(f[:-3]) + ".wav")
                i = i + 1


def create_data_directory_structure():

    os.system("mkdir " + MusicMLConfig.RAW_DATA_DIR)
    os.system("mkdir " + MusicMLConfig.TRAIN_GENRES_SRC)
    os.system("mkdir " + MusicMLConfig.TEST_DATA_DIR)

    for genre_name in MusicMLConfig.GENRE_NAMES:
        os.system("mkdir " + MusicMLConfig.TRAIN_GENRES_SRC + '/' + genre_name)
        os.system("mkdir " + MusicMLConfig.TEST_DATA_DIR + '/' + genre_name)

