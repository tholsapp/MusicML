
class MusicMLConfig:

    #############################
    ### PROJECT CONFIGURATION ###
    #############################

    # MUST CHANGE PER SYSTEM
    ROOT_SRC = "/Users/tholsapp/workstation/MusicML"

    ZIP_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/genres.zip"

    AU_FORMAT_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/au_format/genres"

    AU_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/au_format"

    WAV_FORMAT_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/wav_format"

    TRAINING_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/training"

    TESTING_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/testing"

    UPLOAD_SRC = ROOT_SRC + \
            "/music_ml_backend/resources/data/upload"


    ################################
    ### CLASSIFIER CONFIGURATION ###
    ################################

    # GENRES
    GENRE_LABELS = [
            'blues',
            'classical', 'country',
            'disco',
            'hiphop',
            'jazz',
            'metal',
            'pop',
            'reggae', 'rock',
            ]

    SAMPLE_RATE = 22050  # hertz
    FRAME_SIZE = 2048    # samples
    HOP_SIZE = 512      # samples


    SVC_MODEL_SRC = ROOT_SRC + \
            "/music_ml_backend/classifier/lin_svc_model.pkl"


    ###########################
    ### FLASK CONFIGURATION ###
    ###########################

    FLASK_UPLOAD_DST = ROOT_SRC + \
            "/music_ml_backend/resources/data/upload"

    FLASK_MODEL_SRC =  SVC_MODEL_SRC

