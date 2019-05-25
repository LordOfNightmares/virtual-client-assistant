from entity.entity import Entity
from sql.sqldb import *
from .factoryrepo import FactoryRepo


class DatabaseRepo(FactoryRepo):
    def __init__(self, table):
        import config
        self.db = Sql(config.get_config('database'))
        self.table = table
        super().__init__()

    def add(self, user):
        self.entities[1] = user

    def get(self, id):
        entity = Entity()
        entity.accessed = entity.now()
        del entity.created
        del entity.modified
        entity.id = id
        self.db.commit()
        return [i for i in self.db.select_one(self.table, id)][1:]

    def save(self, entity, timestamp=True):
        # verify if id exists, if exists then update else insert
        if entity.get_id() is None:
            if timestamp:
                entity.created = entity.now()
            self.db.insert(self.table, entity)
        else:
            if timestamp:
                entity.created = __class__.get(self, entity.get_id())[3]
                entity.modified = entity.now()
            self.db.update(self.table, entity)
        self.db.commit()
        # obtain last insert id in self
        entity.id = self.db.last_id(self.table)
        return entity

    def find(self, column, thing, element="*"):
        return self.db.select_one_thing(element, self.table, column, thing)

    def delete(self, entity):
        self.db.delete(self.table, entity)
        self.db.commit()
