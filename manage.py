#!/usr/bin/env python

import os
import sys
from configobj import ConfigObj
from validate import Validator
from flask_script import Manager
import logging

from config import MusicMLConfig
from music_ml_backend.ml.music_ml import \
        test_knn, test_rft, test_svc, test_mlp, \
        extract_and_save_features, test_model, train_and_save_model
from music_ml_backend.util.setup import convert_au_to_wav
from music_ml_frontend.music_ml import app, init_webapp


logging.basicConfig(level=logging.INFO)
music_ml_log = logging.getLogger(__name__)

manager = Manager(app)


@manager.command
def runserver(*args, **kwargs):
    """Initialize and run Flask server

    Overrides default `runserver` to init webapp before running.

    """
    music_ml_log.info("initializing web server")
    app = init_webapp()
    config = ConfigObj('config/sample.config', configspec='config/sample.configspec')
    app.config_obj = config
    app.run(*args, **kwargs)


@manager.command
def setup():
    # convert the raw data into a format we can use
    convert_au_to_wav()


@manager.command
def train_model():
    # convert the raw data into a format we can use
    convert_au_to_wav()

    # extract all features from data
    extract_and_save_features(
            MusicMLConfig.FEATURE_DATASET_SRC,
            MusicMLConfig.FORMATTED_DATA_SRC
            )


@manager.command
def test():
    #test_knn(MusicMLConfig.FEATURE_DATASET_SRC)
    test_rft(MusicMLConfig.FEATURE_DATASET_SRC)
    #test_svc(MusicMLConfig.FEATURE_DATASET_SRC)
    #test_mlp(MusicMLConfig.FEATURE_DATASET_SRC)


if __name__ == "__main__":
    manager.run()

