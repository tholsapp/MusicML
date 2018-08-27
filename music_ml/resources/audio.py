class Audio:
    def __init__(self, src):
        raise NotImplementedError

    def __str__(self):
        return f"<Audio {self!r}>"

    def _open(self, src):
        raise NotImplementedError

