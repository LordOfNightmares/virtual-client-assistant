from .category import Category
from .categoryrepo import CategoryDbRepo
from .databaserepo import DatabaseRepo
from .pattern import Pattern


class PatternDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Ai_stories")

    def save(self, entity):
        # verify if id exists, if exists then update else insert
        query = "INSERT INTO Ai_stories(category_id) VALUES " + "(" + str(entity.category_id) + ")"
        print(query)
        self.db.execute(query)
        for pkey, pval in entity.questions.items():
            entity.id = self.db.last_id(self.table)
            query = "INSERT INTO Ai_qas(question,answer,story_id,facts_id) VALUES " \
                    + "('" + entity.questions[pkey][0] + "','" + entity.questions[pkey][1] + "'," + str(
                entity.id) + "," + str(pkey) + ")"
            print(pkey, query)
            self.db.execute(query)
        for episode_key, episode_val in entity.episodes.items():
            query = "INSERT INTO Ai_episodes(story_id,episode_id,text) VALUES " \
                    + "(" + str(entity.id) + "," + str(episode_key) + ",'" + episode_val + "')"
            self.db.execute(query)
        print("here")
        # else:
        #     self.db.update(self.table, entity)
        self.db.commit()
        # obtain last insert id in self
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
                        p = Pattern(episodes, quas, c.id)
                        quas = {}
                        episodes = {}
                        self.save(p)
                if "?" in vector:
                    question, answer = tuple(vector.split("?", 1))
                    answer, fact_id = tuple(answer.strip().split("\t", 1))
                    quas.update({fact_id: [question, answer]})
                else:
                    episodes.update({vector_id: vector.replace("\n", '')})
                if line is last:
                    p = Pattern(episodes, quas, c.id)
                    self.save(p)

    def upload_glove_to_db(self, glove_vectors_file):
        pass

    def all(self, category_id):
        query = "SELECT * FROM Ai_stories WHERE category_id = '" + str(category_id) + "'"
        stories_results = self.db.select(query)
        all = [self.get(story[0]) for story in stories_results if stories_results]
        return all

    def get_episodes(self, id):
        query = "SELECT * FROM Ai_episodes WHERE story_id = '" + str(id) + "' order by story_id ASC, episode_id ASC"
        episodes = self.db.select(query)
        episodes = [list(i)[2:] for i in episodes]
        eps = {}
        for ep in episodes:
            eps.update({ep[0]: ep[1]})
        return eps

    def get_qas(self, id):
        query = "SELECT * FROM Ai_qas WHERE story_id = '" + str(id) + "'"
        qas = [list(qa) for qa in self.db.select(query)]
        questions, answers, fact_ids = [], [], []
        qas_processed = {}
        for qa in qas:
            qas_processed[qa[4]] = [qa[1], qa[2]]
            # questions.append(qa[1])
            # answers.append(qa[2])
            # fact_ids.append(qa[4])
        return qas_processed

    def get(self, id):
        p = Pattern(self.get_episodes(id), self.get_qas(id), id=id)
        print(p)
        return p
