#!/usr/bin/env python

import os
import sys

from configobj import ConfigObj
from validate import Validator
from flask_script import Manager
import logging

from config import MusicMLConfig
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
    app = init_webapp()
    config = ConfigObj('config/sample.config', configspec='config/sample.configspec')
    app.config_obj = config
    app.run(*args, **kwargs)

@manager.command
def setup():
    create_data_directory_structure()

    convert_au_to_wav(MusicMLConfig.RAW_DATA_DIR)


if __name__ == "__main__":
    manager.run()
