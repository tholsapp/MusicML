"""
This module preprocesses, trains and classifies music genres
"""
import argparse
import logging
import itertools
import joblib
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
from music_ml_backend.classifier.genres import extract_features, extract_sample

from music_ml_backend.resources.audio_manager import AudioManager

log = logging.getLogger(__name__)

normalization_min_max = [(2.1294889e-05, 0.5496966), (2.0222984e-05, 0.42632255), (0.18900083360391381, 0.2858949202325297), (0.0399678360506149, 0.21820178328867756), (0.2976007262690506, 0.7590649887073092), (0.15849518885886182, 0.3223830330513651), (0.021821980133410673, 0.3221444043648492), (0.005700070360419424, 0.1717454117554185), (553.6232545189703, 5413.970526670751), (54.3841387400816, 1830.2904484975766), (12.117722681717163, 28.60416731692146), (2.726628430801156, 11.690313039254097), (705.0170636053144, 3627.692841941549), (55.10536870641002, 1163.9775230854552), (713.4434817285382, 9275.264872752321), (35.5451560969011, 3105.270912813102), (-3.387747575658032, 215.63082127782016), (4.831899195453029, 69.1727758702849), (-92.01253178626331, 65.9392189256682), (2.362758336294578, 55.09747334779238), (-26.25682140001378, 81.82347736619387), (1.9940526506913316, 31.62377110362099), (-40.82934214549017, 36.92069674165085), (4.376576805239685, 32.81706027674665), (-31.017463564471313, 49.30020754255009), (2.718414193398198, 23.686226270250494), (-33.29706058333179, 24.24220388380575), (2.883063484646976, 21.041741147104588), (-33.507124521708455, 48.12342019000106), (3.613692989661358, 24.495753991301157), (-33.701043620399396, 16.412118440921073), (3.494645043511889, 18.755996754805803), (-17.3579124463362, 40.3858459838079), (3.5366198659712014, 23.08346371451842), (-30.341753869604712, 19.50380748346011), (2.336066202954726, 26.944659872089982), (-17.884221507631437, 24.60783328159572), (1.7362359726246084, 27.984830045024246), (-28.671199385200257, 16.771758344765665), (2.9401731447508834, 22.06751079048744), (-17.13050497794845, 23.422976206568276), (2.1393530239817786, 23.842345926557815)]


def genre_classifier(model_names, training_src, testing_src):
    """
    Train and test a model to classify  music genres

    genres:
        [blues, classical, country, blues, hiphop, jazz]

    :param training_src:
    :param testing_src:
    :return:
    """
    valid_model_names = ['SVC', 'MLP', 'KNN']

    if all(model_name in valid_model_names for model_name in model_names):
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

        log.info(f"Min and Maxes for Normalization : {features_min_max}")

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


        train_x = [d[1] for d in n_training_features]
        train_y = [d[0] for d in n_training_features]
        test_x = [d[1] for d in n_testing_features]
        test_y = [d[0] for d in n_testing_features]

        for model_name in model_names:

            if model_name == 'SVC':
                test_svm_model(train_x, train_y, test_x, test_y)

            elif model_name == 'MLP':
                test_mlp_model(train_x, train_y, test_x, test_y)

            elif model_name == 'KNN':
                test_knn_model(train_x, train_y, test_x, test_y)


def test_mlp_model(train_x, train_y, test_x, test_y):
    log.info("Training and Testing MLP Model")

    mlp = MLPClassifier(max_iter=500,random_state=2,hidden_layer_sizes=[40,40])
    mlp.fit(train_x,train_y)

    test_model(mlp, train_x, train_y, test_x, test_y, "MLP")

    joblib.dump(mlp, "mlp_model.pkl")


def test_svm_model(train_x, train_y, test_x, test_y):
    log.info("Training and Testing SVM Model")

    svm = SVC(kernel="linear", C=1.0, random_state=2)
    svm.fit(train_x,train_y)

    test_model(svm, train_x, train_y, test_x, test_y, "LINER SVM")

    joblib.dump(svm, "lin_svc_model.pkl")


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


def make_prediction(file_src):
    model = joblib.load(MusicMLConfig.SVC_MODEL_SRC)
    features = extract_sample(f"{MusicMLConfig.UPLOAD_SRC}/{file_src}")
    nfeatures = []
    for i in range(len(normalization_min_max)):
         denom = normalization_min_max[i][1] - normalization_min_max[i][0]
         if denom == 0:
             nfeatures.append(features[i])
         else:
             nfeatures.append(features[i] - normalization_min_max[i][0] / denom)

    return model.predict([nfeatures])[0]


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
