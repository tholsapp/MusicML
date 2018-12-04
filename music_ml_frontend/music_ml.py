import os

from flask import Flask, request, redirect, url_for, render_template, \
        flash, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from config import MusicMLConfig
from music_ml_backend.classifier.classifier import make_prediction

ALLOWED_EXTENSIONS = set(['wav'])


# Initialize Flask
app = Flask(__name__)


def init_webapp():
    """Initialize the web application."""
    # Initialize Flask-Bootstrap
    Bootstrap(app)

    # Set upload location
    app.config['UPLOAD_FOLDER'] = MusicMLConfig.FLASK_UPLOAD_DST
    app.config['SECRET_KEY'] = 'abc'
    return app


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('idex.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            actual_genre = request.form['genres']
            return redirect(url_for('classify',
                filename=filename,
                actual_genre=actual_genre))

    return render_template('index.html')


@app.route('/classify/<filename>&<actual_genre>')
def classify(filename, actual_genre):
    # classify song
    predicted_genre = make_prediction(filename)

    return render_template('classifier.html',
            filename=filename,
            actual_genre=actual_genre,
            predicted_genre=predicted_genre)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

