"""
Audio Feature Extractor
"""
import logging
import librosa
import numpy as np

from config import MusicMLConfig
from music_ml_backend.resources.audio_manager import AudioManager

log = logging.getLogger(__name__)


def extract_features(
        genre_dir_src,
        sample_rate=MusicMLConfig.SAMPLE_RATE,
        frame_size=MusicMLConfig.FRAME_SIZE,
        hop_size=MusicMLConfig.HOP_SIZE):
    log.info(f"Extracting features from directory {genre_dir_src}")

    audio_genre_map = AudioManager.get_genre_map(genre_dir_src);

    genre_feature_data = []

    duration = 10.0
    for genre in audio_genre_map:
        for audio in audio_genre_map[genre]:
            offset = 0.0
            # increases dataset by factor of 4
            for i in range(2):
                sample, sr = librosa.load(
                        audio.src,
                        offset=offset,
                        sr=sample_rate,
                        duration=duration)
                features = _extract_features(
                        sample,
                        sample_rate,
                        frame_size,
                        hop_size)
                genre_feature_data.append(
                        (genre, features)
                        )
                offset += 15.0
            #break
    return genre_feature_data


def extract_sample(file_src):
    sample, sr = librosa.load(
            file_src,
            offset=0.0,
            sr=MusicMLConfig.SAMPLE_RATE,
            duration=10.0)
    features = _extract_features(
                sample,
                MusicMLConfig.SAMPLE_RATE,
                MusicMLConfig.FRAME_SIZE,
                MusicMLConfig.HOP_SIZE)
    return features


def _extract_features(sample, sample_rate, frame_size, hop_size):
    """
    param sample : audio time series
    param sample_rate : number > 0
                        audio sampling rate of `y`
    param frame_size : Length of the frame
    param step_size : number of samples between successive chroma frames
    """

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=sample, frame_length=frame_size, hop_length=hop_size)

    spectral_centroid = librosa.feature.spectral_centroid(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    spectral_contrast = librosa.feature.spectral_contrast(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    spectral_rolloff = librosa.feature.spectral_rolloff(
            y=sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    mfccs = librosa.feature.mfcc(
                y=sample, sr=sample_rate, n_fft=frame_size,
                hop_length=hop_size)

    flatness = librosa.feature.spectral_flatness(y=sample)
    chroma_cens = librosa.feature.chroma_cens(y=sample,sr=sample_rate)
    chroma_cqt = librosa.feature.chroma_cqt(y=sample, sr=sample_rate)

    features = [
            np.mean(flatness),
            np.std(flatness),
            np.mean(chroma_cens),
            np.std(chroma_cens),
            np.mean(chroma_cqt),
            np.std(chroma_cqt),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_contrast),
            np.std(spectral_contrast),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            ]

    for i in range(13):
        features.append(np.mean(mfccs[i + 1, :]))
        features.append(np.std(mfccs[i + 1, :]))

    return features

