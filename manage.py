#!/usr/bin/env python

from configobj import ConfigObj
from flask_script import Manager
import logging
import os
import sys

from config import MusicMLConfig
from music_ml_backend.classifier.classifier import genre_classifier
from music_ml_frontend.music_ml import app, init_webapp
from music_ml_setup import MusicMLSetup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

manager = Manager(app)


def my_timer(orig_func):
    import time
    def wrapper(*args, **kwargs):
        t1 = time.time()
        rv = orig_func(*args, **kwargs)
        t2 = time.time()
        et = t2 - t1
        print(f"Elapsed Time : {et/60:.0f} mins {et%60:.0f} secs")
        return rv
    return wrapper


@manager.command
def runserver(*args, **kwargs):
    """Initialize and run Flask server

    Overrides default `runserver` to init webapp before running.

    """
    @my_timer
    def in_runserver():
        log.info("initializing web server")
        app = init_webapp()
        config = ConfigObj('config/sample.config', configspec='config/sample.configspec')
        app.config_obj = config
        app.run(*args, **kwargs)

        return None

    return in_runserver()


@manager.command
def setup():
    """
    """
    @my_timer
    def in_setup():
        # sets up project directory structure
        MusicMLSetup.create_project_structure()
        # Extract features, train, test, and save model
        #classifier = genre_classifier("SVC", MusicMLConfig.TRAINING_SRC, MusicMLConfig.TESTING_SRC)
        # Save Model
        # save(classifier)
        return None

    return in_setup()


@manager.command
def test():
    """
    """
    @my_timer
    def in_test():
        MusicMLSetup.create_project_structure()
        # Traing and test Model
        genre_classifier(
                ["SVC", "MLP", "KNN"],
                MusicMLConfig.TRAINING_SRC,
                MusicMLConfig.TESTING_SRC)
        return None

    return  in_test()


@manager.command
def shuffle():
    """
    """
    @my_timer
    def in_shuffle():
        MusicMLSetup.create_project_structure()
        # shuffle training and testing data
        MusicMLSetup.shuffle_training_testing()
        return None

    return in_shuffle()


if __name__ == "__main__":
    manager.run()

