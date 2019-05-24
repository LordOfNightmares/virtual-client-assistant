from .entity import Entity


class Category(Entity):
    def __init__(self, title, id=None):
        self.id = id
        self.title = title
