#!/usr/bin/env python

import os
import sys
from configobj import ConfigObj
from validate import Validator
from flask_script import Manager
import logging

from config import MusicMLConfig
from music_ml_backend.ml.music_ml import \
        test_knn_model, extract_and_save_features, test_model, train_and_save_model
from music_ml_backend.util.setup import convert_au_to_wav, \
        create_data_directory_structure
from music_ml_frontend import app, init_webapp


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

manager = Manager(app)


@manager.command
def runserver(*args, **kwargs):
    """Initialize and run Flask server

    Overrides default `runserver` to init webapp before running.

    """
    log.info("initializing web server")
    app = init_webapp()
    config = ConfigObj('config/sample.config', configspec='config/sample.configspec')
    app.config_obj = config
    app.run(*args, **kwargs)


@manager.command
def setup():
    # set up directory structure for project
    create_data_directory_structure()
    # convert the raw data into a format we can use
    # NOTE: system requires `sox` command
    convert_au_to_wav(MusicMLConfig.RAW_DATA_DIR)


@manager.command
def train_model():
    # set up directory structure for project
    #create_data_directory_structure()
    # convert the raw data into a format we can use
    # NOTE: system requires `sox` command
    #convert_au_to_wav(MusicMLConfig.RAW_DATA_DIR)
    #
    # extract training features
    extract_and_save_features(
            MusicMLConfig.FEATURE_DATASET_SRC,
            MusicMLConfig.TRAIN_GENRES_SRC
            )

    # extract testing features
    extract_and_save_features(
            MusicMLConfig.TEST_FEATURES_SRC,
            MusicMLConfig.TEST_DATA_DIR
            )

    train_and_save_model(
            MusicMLConfig.MODEL_SRC,
            MusicMLConfig.FEATURE_DATASET_SRC
            )

@manager.command
def t():
    test_knn_model(
            MusicMLConfig.FEATURE_DATASET_SRC,
            MusicMLConfig.TEST_FEATURES_SRC
            )


@manager.command
def test():
    test_model(
            MusicMLConfig.MODEL_SRC,
            MusicMLConfig.TEST_DATA_DIR
            )

if __name__ == "__main__":
    manager.run()

