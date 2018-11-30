
class MusicMLConfig:

    # PATHS

    # MUST CHANGE WHEN EXPORTED
    ROOT_SRC = "/Users/tholsapp/sjsu/MusicML"

    UPLOAD_DST = ROOT_SRC + "/music_ml_backend/resources/data/upload"

    RAW_DATA_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/raw/genres"
    FORMATTED_DATA_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/formatted/genres"
    FEATURE_DATASET_SRC = ROOT_SRC + \
            "/music_ml_backend/ml/data/audio_features.csv"

    TRAINING_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/training"

    VALIDATION_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/validation"

    TESTING_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/testing"


    FLASK_FEATURE_DATASET_SRC = FEATURE_DATASET_SRC

    FLASK_MODEL_SRC =  ROOT_SRC + \
            "/music_ml_backend/ml/data/flask_model.pkl"

    KNN_MODEL_SRC = ROOT_SRC + \
            "/music_ml_backend/ml/data/knn_model.pkl"

    RFT_MODEL_SRC = ROOT_SRC + \
            "/music_ml_backend/ml/data/rft_model.pkl"

    SVC_MODEL_SRC = ROOT_SRC + \
            "/music_ml_backend/ml/data/svc_model.pkl"

    MLP_MODEL_SRC = ROOT_SRC + \
            "/music_ml_backend/ml/data/mlp_model.pkl"


    # MUSICML SEETINGS
    GENRE_LABELS = [
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
            'MEAN_SPEC_ROLL_OFF', 'STD_SPEC_ROLL_OFF',
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

