from music_ml_backend.exceptions.base_exception import MMLException


class AudioException(MMLException):
    def __init__(self, message, errors=None):
        MMLException.__init__(self, message=message, errors=errors)

class GraphException(MMLException):
    def __init__(self, message, errors=None):
        MMLException.__init__(self, message=message, errors=errors)

