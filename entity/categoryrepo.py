
from .databaserepo import DatabaseRepo


class CategoryDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Ai_categories")

    def save(self, entity):
        # verify if id exists, if exists then update else insert

        if entity.get_id() is None:
            query = "INSERT INTO Ai_categories(title) VALUES " + "('" + entity.title[len("./neural/tasks_1-20_v1-2/en/"):]+"')"
            self.db.execute(query)
        self.db.commit()
        entity.id = self.db.last_id(self.table)
        return entity

    # def dataset_path(self):
    #     import os
    #     categories = []
    #     for file in os.listdir("./neural/tasks_1-20_v1-2/en/"):
    #         categories.append("./neural/tasks_1-20_v1-2/en/" + file)
    #     return categories
