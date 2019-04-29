from .entity import Entity


class Policy(Entity):
    def __init__(self, title, body, user_id, id=None, hits=0, misses=0):
        self.id = id
        self.title = title
        self.body = body
        self.user_id = user_id
        self.hits = hits
        self.misses = misses
        super().__init__()

    # def __delete__(self, id):
    #     return self.id

    def get_title(self):
        return self.title

    def get_body(self):
        return self.body

    def get_user_id(self):
        return self.user_id

    def get_hits(self):
        return self.hits

    def get_misses(self):
        return self.misses
