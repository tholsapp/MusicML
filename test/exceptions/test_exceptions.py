import pytest

from music_ml_backend.exceptions.base_exception import MMLException
from music_ml_backend.exceptions.exceptions import AudioException, GraphException


@pytest.fixture
def base_exception():
    return MMLException("This is a base exception")


@pytest.fixture
def audio_exception():
    return AudioException("This is an audio exception")


@pytest.fixture
def graph_exception():
    return GraphException("This is a graph exception")


def test_mml_exception(base_exception):
    with pytest.raises(MMLException, message="Expecting MMLException") as exc_info:
        raise base_exception
    assert exc_info.value.message == "This is a base exception"
    assert exc_info.value.errors == None


def test_audio_exception(audio_exception):
    with pytest.raises(AudioException, message="Expecting AudioException") as exc_info:
        raise audio_exception
    assert exc_info.value.message == "This is an audio exception"
    assert exc_info.value.errors == None


def test_graph_exception(graph_exception):
    with pytest.raises(GraphException, message="Expecting GraphException") as exc_info:
        raise graph_exception

    assert exc_info.value.message == "This is a graph exception"
    assert exc_info.value.errors == None

