from .entity import Entity


class Embedding(Entity):
    def __init__(self, word, vector, id=None):
        self.id = id
        self.word = word
        self.vector = vector

    def get_id(self):
        return self.id
