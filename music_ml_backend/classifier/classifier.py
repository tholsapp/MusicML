"""
A perceptron classifier

This module preprocesses, trains and classifies music genres

"""
import argparse
import logging
import itertools
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# required for my system
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn
import numpy as np

from config import MusicMLConfig
from music_ml_backend.classifier.genres import extract_features

from music_ml_backend.resources.audio_manager import AudioManager

log = logging.getLogger(__name__)


def genre_classifier(training_src, testing_src):
    """
    Train and test a model to classify  music genres

    genres:
        [blues, classical, country, blues, hiphop, jazz]

    :param training_src:
    :param testing_src:
    :return:
    """
    training_features = extract_features(training_src)
    testing_features = extract_features(testing_src)

    random.shuffle(training_features)
    random.shuffle(testing_features)

    log.info("Normalizing Features")

    feature_vectors = []
    for feature_vector in training_features:
        feature_vectors.append(feature_vector[1])
    for feature_vector in testing_features:
        feature_vectors.append(feature_vector[1])

    n_training_features = []
    n_testing_features = []

    features_min_max = []
    for i in range(len(feature_vectors[0])):
        fmin = 2**64
        fmax = -(2**64)
        for vector in feature_vectors:
            if vector[i] > fmax:
                fmax = vector[i]
            if vector[i] < fmin:
                fmin = vector[i]
        log.info(f" Feature : {i:3.0f}, Min : {fmin:8.2f}, Max : {fmax:8.2f}")
        features_min_max.append((fmin, fmax))

    #log.info(f"Min and Maxes for Normalization : {features_min_max}")

    for t in training_features:
        feature_vector = [0 for x in features_min_max]
        for i in range(len(features_min_max)):
            denom = features_min_max[i][1] - features_min_max[i][0]
            if denom == 0:
                feature_vector[i] = t[1][i]
            else:
               feature_vector[i] = \
                       (t[1][i] - features_min_max[i][0]) / denom
        n_training_features.append((t[0], feature_vector))

    for t in testing_features:
        feature_vector = [0 for x in features_min_max]
        for i in range(len(features_min_max)):
            denom = features_min_max[i][1] - features_min_max[i][0]
            if denom == 0:
                feature_vector[i] = t[1][i]
            else:
               feature_vector[i] = \
                       (t[1][i] - features_min_max[i][0]) / denom
        n_testing_features.append((t[0], feature_vector))

    test_svm_model(
            [d[1] for d in n_training_features],
            [d[0] for d in n_training_features],
            [d[1] for d in n_testing_features],
            [d[0] for d in n_testing_features])

    #test_mlp_model(
    #        [d[1] for d in n_training_features],
    #        [d[0] for d in n_training_features],
    #        [d[1] for d in n_testing_features],
    #        [d[0] for d in n_testing_features])

    #test_knn_model(
    #        [d[1] for d in n_training_features],
    #        [d[0] for d in n_training_features],
    #        [d[1] for d in n_testing_features],
    #        [d[0] for d in n_testing_features])


def test_mlp_model(train_x, train_y, test_x, test_y):
    log.info("Training and Testing MLP Model")

    mlp = MLPClassifier(max_iter=500,random_state=2,hidden_layer_sizes=[40,40])
    mlp.fit(train_x,train_y)

    test_model(mlp, train_x, train_y, test_x, test_y, "MLP")


def test_svm_model(train_x, train_y, test_x, test_y):
    log.info("Training and Testing SVM Model")

    svm = SVC(kernel="linear", C=1.0, random_state=2)
    svm.fit(train_x,train_y)

    test_model(svm, train_x, train_y, test_x, test_y, "LINER SVM")

    poly_svm = SVC(kernel="poly", degree=2, C=1.0, random_state=2)
    poly_svm.fit(train_x,train_y)

    test_model(poly_svm, train_x, train_y, test_x, test_y, "POLY SVM")

    rbf_svm = SVC(kernel="rbf", C=1.0, random_state=2)
    rbf_svm.fit(train_x,train_y)

    test_model(rbf_svm, train_x, train_y, test_x, test_y, "RBF SVM")


def test_knn_model(train_x, train_y, test_x, test_y):
    log.info("Training and Testing KNN Model")
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

    test_model(knn, train_x, train_y, test_x, test_y, "KNN")


def test_model(model, train_x, train_y, test_x, test_y, model_name=""):

    pred_labels = model.predict(test_x)

    print("\n-----------------------------------------------------------------")
    print(f"---------------PERFORMANCE ANALYSIS FOR THE {model_name} MODEL--------\n")

    print_confusion_matrix(
            MusicMLConfig.GENRE_LABELS,
            pred_labels,
            test_y,
            model_name)

    print(f"Training Score : {model.score(train_x,train_y):.3f}")
    print(f"Test Score     : {model.score(test_x,test_y):.3f}")

    print("-----------------------------------------------------------------\n")

    #matrix=sklearn.metrics.confusion_matrix(test_y, pred_labels)
    #plt.figure()
    #a=confusion_matrix(matrix,classes=MusicMLConfig.GENRE_NAMES, title=f"{model_name} Confusion matrix")


def print_confusion_matrix(labels, pred_labels, true_labels, model_name):

    matrix = {l : [0 for i in labels] for l in labels}
    lab_pos = {l : 0 for l in labels}
    for i in range(len(labels)):
        lab_pos[labels[i]] = i
    for i in range(len(true_labels)):
        matrix[true_labels[i]][lab_pos[pred_labels[i]]] += 1

    print("----------------------------------------------------------")
    print(f"{model_name} Confusion Matrix")
    print(f"            ", end=" ")
    for l in matrix:
        print(f"{l[:3]:3} ", end=" ")
    print()

    for l in matrix:
        print(f"{l:9} : ", end=" ")
        for v in matrix[l]:
            print(f"{str(v):4}", end=" ")
        print()
    print("----------------------------------------------------------\n")


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
    plt.show()
