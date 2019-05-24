from entity.conversation import Conversation
from .databaserepo import DatabaseRepo


class ConversationDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Conversations")

    def get(self, id):
        r_select = self.db.select_one(self.table, id)
        if r_select:
            # print('crg',r_select[3])
            # current = [i for i in self.db.select_one(self.table, id)]
            m = Conversation(id, r_select[3])
            return m
        else:
            return None

    def save(self, entity):
        # verify if id exists, if exists then update else insert
        found_entity = self.get(entity.get_id())
        if found_entity is None:
            entity.created = entity.now()
            self.db.insert(self.table, entity)
        else:
            # entity.created = __class__.get(self, entity.get_id())[3]
            entity.modified = entity.now()
            self.db.update(self.table, entity)
        self.db.commit()
        # obtain last insert id in self
        entity.id = self.db.last_id(self.table)
        return entity
