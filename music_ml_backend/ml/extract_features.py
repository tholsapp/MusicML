import logging
import librosa
import numpy as np
import pandas as pd
import sklearn

from config import MusicMLConfig
from music_ml_backend.resources.audio_manager import AudioManager

log = logging.getLogger(__name__)


def extract_normalized_features(
        genres_src,
        sample_rate=MusicMLConfig.SAMPLE_RATE,
        frame_size=MusicMLConfig.FRAME_SIZE,
        hop_size=MusicMLConfig.HOP_SIZE):

    audio_genre_map = AudioManager.get_genre_map(genres_src);
    log.info("Beginning to extract features")
    labels = []
    is_created = False

    for genre in audio_genre_map:
        print(genre)
        for audio in audio_genre_map[genre]:
            # gets sample of this audio src
            labels.append(genre)
            sample, sr = librosa.load(audio.src, sr=sample_rate, duration=5.0)
            features = _extract_features(
                    sample, sample_rate, frame_size, hop_size)
            if not is_created:
                np_dataset = np.array(features)
                is_created = True
            elif is_created:
                np_dataset = np.vstack((np_dataset, features))

    np_normalized_dataset = _normalize_dataset(np_dataset)

    feature_df = pd.DataFrame(np_dataset, columns=MusicMLConfig.FEATURE_NAMES)

    feature_df[MusicMLConfig.LABEL_NAME] = labels;

    return feature_df


def extract_normalized_featuress(
        genres_src,
        sample_rate=MusicMLConfig.SAMPLE_RATE,
        frame_size=MusicMLConfig.FRAME_SIZE,
        hop_size=MusicMLConfig.HOP_SIZE):

    audio_genre_map = AudioManager.get_genre_map(genres_src);
    log.info("Beginning to extract features")
    labels = []
    is_created = False

    for genre in audio_genre_map:
        print(genre)
        for audio in audio_genre_map[genre]:
            # gets sample of this audio src
            labels.append(genre)
            sample, sr = librosa.load(audio.src, sr=sample_rate, duration=20.0)
            features = _extract_features(
                    sample, sample_rate, frame_size, hop_size)
            if not is_created:
                np_dataset = np.array(features)
                is_created = True
            elif is_created:
                np_dataset = np.vstack((np_dataset, features))

    np_normalized_dataset = _normalize_dataset(np_dataset)

    feature_df = pd.DataFrame(np_dataset, columns=MusicMLConfig.FEATURE_NAMES)

    return feature_df


def _extract_features(sample, sample_rate, frame_size, hop_size):
    """
    param ats : audio time series
    param sample_rate : number > 0
                        audio sampling rate of `y`
    param frame_size : Length of the frame
    param step_size : number of samples between successive chroma frames
    """

    log.info("Extracting zero crossing rate")
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=sample, frame_length=frame_size, hop_length=hop_size)

    log.info("Extracting spectral centroid")
    spectral_centroid = librosa.feature.spectral_centroid(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    log.info("Extracting spectral contrast")
    spectral_contrast = librosa.feature.spectral_contrast(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    log.info("Extracting spectral bandwidth")
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    log.info("Extracting spectral rolloff")
    spectral_rolloff = librosa.feature.spectral_rolloff(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    log.info("Extracting mel-frequency cepstrum")
    mfccs = librosa.feature.mfcc(
                y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    ret_lst = [
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_contrast),
            np.std(spectral_contrast),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
            ]

    for i in range(13):
        ret_lst.append(np.mean(mfccs[i + 1, :]))
        ret_lst.append(np.std(mfccs[i + 1, :]))

    return ret_lst

def _normalize_dataset(np_dataset):
    log.info("Normalizing the data")
    # range = (-1, 1) since audio is a wave form
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return min_max_scaler.fit_transform(np_dataset)

