class FactoryRepo:
    def __init__(self):
        self.entities = {}

    def add(self, entity):
        self.entities[entity.getID()] = entity

    def all(self):
        return self.entities

    def get(self, id):
        return self.entities[id]
