import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import MusicMLConfig
from music_ml_backend.util.ml_util import read_features

log = logging.getLogger(__name__)


def train_model(csv_src):
    np_dataset = np.array(read_features(csv_src))

    number_of_rows, number_of_cols = np_dataset.shape

    x = np_dataset[:, :number_of_cols - 1]
    y = np_dataset[:, number_of_cols - 1]

    model = SVC(C=100, gamma=0.08)
    log.info("Training the model")
    model.fit(x, y)

    return model

