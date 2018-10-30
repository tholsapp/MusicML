
class MusicMLConfig:

    ROOT_DIR = "/Users/tholsapp/workstation/sjsu/cs161/MusicML"
    RAW_DATA_DIR = ROOT_DIR + "/music_ml_backend/resources/data/raw/genres"
    TRAIN_GENRES_SRC = ROOT_DIR + "/music_ml_backend/resources/data/training"
    TEST_DATA_DIR = ROOT_DIR + "/music_ml_backend/resources/data/test"

    FEATURE_DATASET_SRC = ROOT_DIR + \
            "/music_ml_backend/ml/data/training_features.csv"

    TEST_FEATURES_SRC = ROOT_DIR + \
            "/music_ml_backend/ml/data/test_features.csv"

    MODEL_SRC = ROOT_DIR + \
            "/music_ml_backend/ml/data/model.pkl"

    GENRE_NAMES = [
            'blues',
            'classical',
            'country',
            'disco',
            'hiphop',
            'jazz',
            'metal',
            'pop',
            'reggae',
            'rock',
            ]

    SAMPLE_RATE = 22050  # hertz
    FRAME_SIZE = 2048    # samples
    HOP_SIZE = 512      # samples

    LABEL_NAME = "GENRE"

    FEATURE_NAMES =  [
            'MEAN_ZCR', 'STD_ZCR',
            'MEAN_SPEC_CENTROID_', 'STD_SPEC_CENTROID_',
            'MEAN_SPEC_CONTRAST_', 'STD_SPEC_CONTRAST_',
            'MEAN_SPEC_BANDWIDTH_', 'STD_SPEC_BANDWIDTH_',
            'MEAN_SPEC_ROLL_OF_', 'STD_SPEC_ROLL_OF_',
            'MEAN_MFCC_1', 'STD_MFCC_1',
            'MEAN_MFCC_2', 'STD_MFCC_2',
            'MEAN_MFCC_3', 'STD_MFCC_3',
            'MEAN_MFCC_4', 'STD_MFCC_4',
            'MEAN_MFCC_5', 'STD_MFCC_5',
            'MEAN_MFCC_6', 'STD_MFCC_6',
            'MEAN_MFCC_7', 'STD_MFCC_7',
            'MEAN_MFCC_8', 'STD_MFCC_8',
            'MEAN_MFCC_9', 'STD_MFCC_9',
            'MEAN_MFCC_10', 'STD_MFCC_10',
            'MEAN_MFCC_11', 'STD_MFCC_11',
            'MEAN_MFCC_12', 'STD_MFCC_12',
            'MEAN_MFCC_13', 'STD_MFCC_13'
            ]

