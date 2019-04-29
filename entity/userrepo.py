from entity.user import User
from .databaserepo import DatabaseRepo


class UserDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Users")

    def get(self, id):
        current = super().get(id)
        u = User(*current[:-3], id)
        u.created, u.modified, u.accessed = current[3], current[4], current[5]
        return u

    def find(self, entity):
        res = self.db.select_where(self.table, entity)
        if res:
            return User(*res[1:-3], *res[1])
        else:
            return None
