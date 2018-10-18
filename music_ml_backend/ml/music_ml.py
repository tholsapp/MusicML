import logging
import joblib

from pandas import DataFrame

from config import MusicMLConfig

from music_ml_backend.util.ml_util import save_features, save_model
from music_ml_backend.ml.extract_features import extract_normalized_features, \
        extract_normalized_featuress
from music_ml_backend.ml.train_model import train_model
import itertools
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from music_ml_backend.util.ml_util import read_features
import numpy as np

log = logging.getLogger(__name__)


def extract_and_save_features(feature_src, data_src):
    save_features(feature_src, _extract_features(data_src))
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


def test_knn_model(train_src, test_src):

    np_dataset = np.array(read_features(train_src))
    test_np_dataset = np.array(read_features(test_src))

    number_of_rows, number_of_cols = np_dataset.shape

    train_x = np_dataset[:, :number_of_cols - 1]
    train_y = np_dataset[:, number_of_cols - 1]

    test_x = test_np_dataset[:, :number_of_cols - 1]
    test_y = test_np_dataset[:, number_of_cols - 1]

    results_knn=[]
    for i in range(1,11):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x,train_y)
        results_knn.append(knn.score(test_x,test_y))

    max_accuracy_knn=max(results_knn)
    best_k=1+results_knn.index(max(results_knn))
    print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn,best_k))

    plt.plot(np.arange(1,11),results_knn)
    plt.xlabel("n Neighbors")
    plt.ylabel("Accuracy")


    knn=KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_x,train_y)
    #return knn
    knn=KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_x,train_y)
    print("Training Score: {:.3f}".format(knn.score(train_x,train_y)))
    print("Test score: {:.3f}".format(knn.score(test_x,test_y)))

    plot_cnf(knn,test_x,test_y,MusicMLConfig.GENRE_NAMES)

    svm=SVC(C=100,gamma=0.08)
    svm.fit(train_x,train_y)
    print("Training Score: {:.3f}".format(svm.score(train_x,train_y)))
    print("Test score: {:.3f}".format(svm.score(test_x,test_y)))

    plot_cnf(svm,test_x,test_y,MusicMLConfig.GENRE_NAMES)


    neural=MLPClassifier(max_iter=400,random_state=2,hidden_layer_sizes=[40,40])
    neural.fit(train_x,train_y)
    print("Training Score: {:.3f}".format(neural.score(train_x,train_y)))
    print("Test score: {:.3f}".format(neural.score(test_x,test_y)))

    plot_cnf(neural,test_x,test_y,MusicMLConfig.GENRE_NAMES)


def plot_cnf(model,dataset_x,dataset_y,GENRES):
    true_y=dataset_y
    true_x=dataset_x
    pred=model.predict(true_x)

    print("---------------PERFORMANCE ANALYSIS FOR THE MODEL----------------\n")

    print("Real Test dataset labels: \n{}\n".format(true_y))
    print("Predicted Test dataset labels: \n{}".format(pred))

    cnf_matrix=sklearn.metrics.confusion_matrix(true_y,pred)
    plt.figure()
    a=confusion_matrix(cnf_matrix,classes=GENRES,title='Confusion matrix')


def confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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

def _extract_features(src):
    return extract_normalized_features(src)

