import pytest

from music_ml_backend.exceptions.exceptions import AudioException
from music_ml_backend.resources.audio import Audio


class TestAudio:
    test_audio_file_path = "test/data/test.pop.00000.wav"
    not_a_valid_audio_file_path = "not-a-audio-file"

    def test_valid_audio_initialization(self):
        audio = Audio(self.test_audio_file_path)
        assert type(audio) == Audio
        assert audio.src == self.test_audio_file_path

    def test_invalid_audio_initialization(self):
        with pytest.raises(AudioException, message="Expecting AudioException"):
            Audio(self.not_a_valid_audio_file_path)
        with pytest.raises(AudioException, message="Expecting AudioException"):
            Audio("")

