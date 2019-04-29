from .entity import Entity


class Message(Entity):
    def __init__(self, title, body, user_id, bot_id, conversation_id, id=None):
        self.id = id
        self.title = title
        self.body = body
        self.user_id = user_id
        self.bot_id = bot_id
        self.conversation_id = conversation_id
        super().__init__()

    def __delete__(self, id):
        return self.id

    def get_title(self):
        return self.title

    def get_body(self):
        return self.body

    def get_user_id(self):
        return self.user_id
