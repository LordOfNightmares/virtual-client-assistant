from .category import Category
from .categoryrepo import CategoryDbRepo
from .databaserepo import DatabaseRepo
from .pattern import Pattern


class PatternDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Ai_stories")

    def save(self, entity):
        query = "INSERT INTO Ai_stories(category_id) VALUES " + "(" + str(entity.category_id) + ")"
        self.db.execute(query)
        '''ID integer primary key AUTOINCREMENT,
                story_id integer,
                episode_id integer,
                text varchar(255),
                typo integer'''

        for pkey, pval in entity.episodes.items():
            entity.id = self.db.last_id(self.table)
            query = "INSERT INTO Ai_dataset(story_id,episode_id,text,typo) VALUES " + "(" + str(
                entity.id) + "," + str(pkey) + ",'" + pval[0] + "'," + str(pval[1]) + ")"
            # print(query)
            self.db.execute(query)
        self.db.commit()
        return entity

    def upload_qa_to_db(self, qa_dataset_file):
        from tqdm import tqdm
        def file_len(fname):
            with open(fname, "r", encoding="utf8") as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        c = Category(qa_dataset_file[:-4])
        cdb = CategoryDbRepo()
        cdb.save(c)

        # print(c)

        episodes = {}
        quas = {}
        with open(qa_dataset_file, "r", encoding="utf8") as glove:
            lines = glove.readlines()
            last = lines[-1]
            for line in tqdm(lines, total=file_len(qa_dataset_file)):
                vector_id, vector = tuple(line.split(" ", 1))
                if int(vector_id) == 1:
                    if len(episodes) > 0:
                        p = Pattern(episodes, c.id)
                        episodes = {}
                        self.save(p)
                if "?" in vector:
                    episodes.update({vector_id: [vector.replace("\n", ''), 1]})
                else:
                    episodes.update({vector_id: [vector.replace("\n", ''), 0]})
                if line is last:
                    p = Pattern(episodes, c.id)
                    self.save(p)

    def upload_glove_to_db(self, glove_vectors_file):
        pass

    def all(self, category_id):
        query = "SELECT * FROM Ai_stories WHERE category_id = '" + str(category_id) + "'"
        stories_results = self.db.select(query)
        all = [self.get(story[0]) for story in stories_results if stories_results]
        return all

    def get_dataset(self, id):
        query = "SELECT * FROM Ai_dataset WHERE story_id=" + str(id)
        dataset = self.db.select(query)
        episodes = {}
        # print(dataset)
        for episode in dataset:
            episodes.update({list(episode)[2]: [list(episode)[3], list(episode)[4]]})
        return episodes

    def get(self, id, category_id=0):
        p = Pattern(self.get_dataset(id), category_id, id=id)
        return p
