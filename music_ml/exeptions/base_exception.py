class MMLExeption(Exception):
    def __init__(self, message, errors=None):
        self.message = message
        self.errors = errors

        Exception.__init__(self, message)

