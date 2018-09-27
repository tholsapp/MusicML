#!/usr/bin/env python

from configobj import ConfigObj
from validate import Validator
from flask_script import Manager
import logging

from music_ml_frontend import app, init_webapp


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


manager = Manager(app)


@manager.command
def runserver(*args, **kwargs):
    """Override default `runserver` to init webapp before running."""
    app = init_webapp()
    config = ConfigObj('config/sample.config', configspec='config/sample.configspec')
    app.config_obj = config
    app.run(*args, **kwargs)


if __name__ == "__main__":
    manager.run()
