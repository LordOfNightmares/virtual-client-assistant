from .entity import Entity


class Pattern(Entity):
    def __init__(self, episodes, questions, category_id=0, id=0):
        self.id = id
        self.episodes = episodes
        self.category_id = category_id
        self.questions = questions
