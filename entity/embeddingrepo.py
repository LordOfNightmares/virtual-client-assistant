from .databaserepo import DatabaseRepo


class EmbeddingDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Ai_embeddings")

    def load_to_db(self, entity, rewrite=True):
        # verify if id exists, if exists then update else insert
        try:
            glove = self.find('word', entity.word, element="rowid")
            test = list(glove)[0] is None and len(glove) <= 0
        except:
            test = True
        if test:
            self.db.insert(self.table, entity)
        else:
            if rewrite:
                self.db.update(self.table, entity, list(glove)[0])
            # print(list(glove)[0])
        # obtain last insert id in self
        entity.id = self.db.last_id(self.table)
        return entity

    # def load(self, entity):
    #     # verify if id exists, if exists then update else insert
    #     if entity.get_id():
    #         self.db.insert(self.table, entity)
    #     else:
    #         self.db.update(self.table, entity)
    #     # obtain last insert id in self
    #     entity.id = self.db.last_id(self.table)
    #     return entity

    def save(self, entity):
        self.load_to_db(entity)
        self.db.commit()
        return entity

    def save_db(self):
        self.db.commit()

    def get(self):
        return self.db.select("SELECT * FROM " + self.table)