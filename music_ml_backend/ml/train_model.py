import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import MusicMLConfig
from music_ml_backend.util.ml_util import read_features

log = logging.getLogger(__name__)


def train_model(csv_src):

    feature_dataset = pd.read_csv(csv_src, index_col=False)

    number_of_rows, number_of_cols = feature_dataset.shape

    feature_values = np.array(feature_dataset)

    training_dataset, testing_dataset = train_test_split(
            feature_values,
            test_size=10,
            random_state=2,
            stratify=feature_values[:,number_of_cols-1]
            )

    #np_dataset = np.array(read_features(csv_src))

    #number_of_rows, number_of_cols = np_dataset.shape

    x = training_dataset[:, :number_of_cols - 1]
    y = training_dataset[:, number_of_cols - 1]

    model = SVC(C=100, gamma=0.08)
    log.info("Training the model")
    model.fit(x, y)

    return model

