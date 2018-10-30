import logging
import joblib
import pandas as pd
import numpy as np

from pandas import DataFrame

from config import MusicMLConfig

from music_ml_backend.util.ml_util import save_features, save_model
from music_ml_backend.ml.extract_features import extract_normalized_features
from music_ml_backend.ml.train_model import train_model
import itertools

# required for my system
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from music_ml_backend.util.ml_util import read_features
import numpy as np

log = logging.getLogger(__name__)


def split_dataset(feature_src, test_size, random_state):
    feature_dataset = pd.read_csv(feature_src, index_col=False)
    number_of_rows, number_of_cols = feature_dataset.shape

    feature_values = np.array(feature_dataset)

    training_dataset, testing_dataset = train_test_split(
            feature_values,
            test_size=test_size,
            random_state=random_state,
            stratify=feature_values[:,number_of_cols-1]
            )

    return (training_dataset, testing_dataset, number_of_cols)


def test_knn_model(feature_src):
    log.info("Traning and Testing KNN Model")

    training_dataset, testing_dataset, number_of_cols  = \
            split_dataset(feature_src, 0.85, 2)

    train_x = training_dataset[:,:number_of_cols-1]
    train_y = training_dataset[:,number_of_cols-1]

    test_x = testing_dataset[:,:number_of_cols-1]
    test_y = testing_dataset[:,number_of_cols-1]


    results_knn=[]
    for i in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x, train_y)
        results_knn.append(knn.score(test_x, test_y))

    max_accuracy_knn = max(results_knn)
    best_k = 1 + results_knn.index(max(results_knn))

    log.info(f"Max Accuracy : {max_accuracy_knn:.3f} , with {best_k!r} neighbors")

    knn=KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_x,train_y)
    #return knn

    knn=KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_x,train_y)

    log.info(f"Training Score : {knn.score(train_x,train_y):.3f}")
    log.info(f"Test Score     : {knn.score(test_x,test_y):.3f}")

    test_model(knn, test_x, test_y)


def test_rft_model(feature_src):
    log.info("Traning and Testing RFT Model")

    training_dataset, testing_dataset, number_of_cols = \
            split_dataset(feature_src, 0.20, 2)

    train_x = training_dataset[:,:number_of_cols-1]
    train_y = training_dataset[:,number_of_cols-1]

    test_x = testing_dataset[:,:number_of_cols-1]
    test_y = testing_dataset[:,number_of_cols-1]

    results_rft=[]
    for i in range(2,20):
        rft = RandomForestClassifier(random_state=42 , n_estimators=i)
        rft.fit(train_x, train_y)
        results_rft.append(rft.score(test_x, test_y))

    max_accuracy_rft = max(results_rft)
    best_n_est = 2 + results_rft.index(max(results_rft))

    log.info(f"Max Accuracy : {max_accuracy_rft:.3f} , with {best_n_est!r} estimators.")

    rft = RandomForestClassifier(random_state=42, n_estimators=best_n_est)
    rft.fit(train_x,train_y)

    log.info(f"training score : {rft.score(train_x,train_y):.3f}")
    log.info(f"test score     : {rft.score(test_x,test_y):.3f}")

    test_model(rft, test_x, test_y)


def test_svc_model(feature_src):
    log.info("Training and Testing SVC Model")

    training_dataset, testing_dataset, number_of_cols = \
            split_dataset(feature_src, 0.30, 2)

    train_x = training_dataset[:,:number_of_cols-1]
    train_y = training_dataset[:,number_of_cols-1]

    test_x = testing_dataset[:,:number_of_cols-1]
    test_y = testing_dataset[:,number_of_cols-1]

    svm = SVC(kernel='linear', C=100, gamma=0.1)
    svm.fit(train_x,train_y)

    log.info(f"training score : {svm.score(train_x,train_y):.3f}")
    log.info(f"test score     : {svm.score(test_x,test_y):.3f}")

    test_model(svm, test_x, test_y)


def test_mlp_model(feature_src):
    log.info("Training and Testing MLP Model")

    training_dataset, testing_dataset, number_of_cols = \
            split_dataset(feature_src, 0.30, 2)

    train_x = training_dataset[:,:number_of_cols-1]
    train_y = training_dataset[:,number_of_cols-1]

    test_x = testing_dataset[:,:number_of_cols-1]
    test_y = testing_dataset[:,number_of_cols-1]

    mlp = MLPClassifier(max_iter=500,random_state=2,hidden_layer_sizes=[40,40])
    mlp.fit(train_x,train_y)

    log.info(f"training score : {mlp.score(train_x,train_y):.3f}")
    log.info(f"test score     : {mlp.score(test_x,test_y):.3f}")

    test_model(mlp, test_x, test_y)


def test_model(model, dataset_x, dataset_y):
    true_y = dataset_y
    true_x = dataset_x

    pred = model.predict(true_x)

    print("\n\n-----------------------------------------------------------------")
    print("---------------PERFORMANCE ANALYSIS FOR THE MODEL----------------\n")

    print(f"Expected dataset labels: \n{true_y}\n")
    print(f"Predicted Test dataset labels: \n{pred}\n\n")

    matrix=sklearn.metrics.confusion_matrix(true_y,pred)
    plt.figure()
    a=confusion_matrix(matrix,classes=MusicMLConfig.GENRE_NAMES, title='Confusion matrix')


def confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Spectral):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
