from .entity import Entity


class Pattern(Entity):
    def __init__(self, episodes, category_id, id=None):
        self.id = id
        self.episodes = episodes
        self.category_id = category_id
