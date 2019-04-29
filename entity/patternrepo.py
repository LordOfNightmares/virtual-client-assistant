from entity.pattern import Pattern
from .databaserepo import DatabaseRepo


class PatternDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Patterns")

    def get(self, id):
        current = super().get(id)
        u = Pattern(*current[:-3], id)
        u.created, u.modified, u.accessed = current[3], current[4], current[5]
        return u
