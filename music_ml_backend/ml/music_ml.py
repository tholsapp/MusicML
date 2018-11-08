
import joblib
import itertools
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame


# required for my system
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import MusicMLConfig

from music_ml_backend.ml.extract_features import extract_normalized_features, \
        flask_extract_features
from music_ml_backend.ml.test_model import test_knn_model, test_rft_model, \
        test_svc_model, test_mlp_model
from music_ml_backend.ml.train_model import train_model
from music_ml_backend.util.ml_util import save_features, save_model
from music_ml_backend.util.ml_util import read_features

log = logging.getLogger(__name__)


def classify(uploaded_filename, label):
    src = MusicMLConfig.UPLOAD_DST + '/' + uploaded_filename
    features = flask_extract_features(src, label)

    features = features.drop(columns=['GENRE'])

    for x in features:
        print(str(x) + " " + str(features[x]))

    svm = joblib.load(MusicMLConfig.FLASK_MODEL_SRC)
    pred = svm.predict(features)
    print(pred)

    return pred[0]


def extract_and_save_features(feature_src, data_src):
    save_features(feature_src, extract_normalized_features(data_src))
    log.info("Extracted and saved features successfully!")


def train_and_save_model(model_src, feature_src):
    save_model(model_src, train_model(feature_src))
    log.info("Trained and saved model successfully!")


def test_model(model_src, data_src):
    test_data = extract_normalized_features(data_src)

    #test_data = test_data[:-1]

    print(test_data)

    svm = joblib.load(model_src)
    print("----------------------------------- Predicted Labels -----------------------------------\n")
    preds = svm.predict(test_data)
    print(preds)
    print("")
    print("----------------------------------------------------------------------------------------")


def test_knn(feature_src):
    test_knn_model(feature_src)


def test_rft(feature_src):
    test_rft_model(feature_src)


def test_svc(feature_src):
    test_svc_model(feature_src)


def test_mlp(feature_src):
    test_mlp_model(feature_src)

