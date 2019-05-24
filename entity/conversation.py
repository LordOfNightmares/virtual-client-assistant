from .entity import Entity


class Conversation(Entity):
    def __init__(self, id, user_id, bot_id=0):
        self.id = id
        self.mark = 0
        self.feedback = 0
        self.user_id = user_id
        self.bot_id = bot_id
        super().__init__()
        del self.accessed
        self.ended = 0

    def get_ended(self):
        return self.ended

    def get_mark(self):
        return self.mark

    def get_feedback(self):
        return self.feedback
