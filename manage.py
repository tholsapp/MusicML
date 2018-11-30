#!/usr/bin/env python

from configobj import ConfigObj
import os
from flask_script import Manager
import logging
import sys
import time
from validate import Validator

from config import MusicMLConfig
from music_ml_setup import MusicMLSetup
from music_ml_backend.classifier.classifier import genre_classifier
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
    """
    """
    t1 = time.time()
    # sets up project directory structure
    MusicMLSetup.create_project_structure()
    # convert raw data
    # and seperate data into training, validation, and testing
    MusicMLSetup.convert_and_split_data()

    # Extract features, train, test, and save model
    classifier = genre_classifier(MusicMLConfig.TRAINING_SRC, MusicMLConfig.TESTING_SRC)

    # Save Model
    # save(classifier)

    t2 = time.time()
    et = t2 - t1
    print(f"Elapsed Time : {et/60:.0f} mins {et%60:.0f} secs")


@manager.command
def test():
    # Traing and test Model
    genre_classifier(MusicMLConfig.TRAINING_SRC, MusicMLConfig.TESTING_SRC)


if __name__ == "__main__":
    manager.run()

