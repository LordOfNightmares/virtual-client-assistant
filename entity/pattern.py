from .entity import Entity


class Pattern(Entity):
    def __init__(self, episodes, question, answer, facts_id, category_id=0, id=0):
        self.id = id
        self.episodes = episodes
        self.category_id = category_id
        self.question = question
        self.answer = answer
        self.facts_id = facts_id
