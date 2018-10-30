import logging

from flask import Flask, render_template
from flask_bootstrap import Bootstrap


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


# Initialize Flask
app = Flask(__name__)


def init_webapp():
    """Initialize the web application."""
    # Initialize Flask-Bootstrap
    log.info("Initializing web application")
    Bootstrap(app)

    return app


@app.route('/')
def index():
    return render_template('index.html')

