from entity.embedding import Embedding
from .databaserepo import DatabaseRepo


class EmbeddingDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Ai_embeddings")

    def load_to_db(self, entity, rewrite):
        # verify if id exists, if exists then update else insert
        if entity.get_id() is None:
            self.db.insert(self.table, entity)
        else:
            if rewrite:
                self.db.fast_update(self.table, entity, entity.id)
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


    def get(self):
        return self.db.select("SELECT * FROM " + self.table)

    def upload_embeddings(self, glove_vectors_file, rewrite=True):
        from tqdm import tqdm

        def generator(row):
            row = list(row)
            rowe = []
            for i in range(len(row)):
                row[i] = str(row[i])
                rowe.append([])
                for j in range(len(row[i])):
                    rowe[i].append(row[i][j])
                    if row[i][j] == "\'":
                        rowe[i].append("\'")
            row = [''.join(rowe[i]) for i in range(len(rowe))]

            string = "".join(''.format(k) for k in row)
            return string

        def word_chck(word):
            if "\'" in word:
                # print(word)
                word = generator(word)
            return word

        def file_len(fname):
            with open(fname, "r", encoding="utf8") as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        with open(glove_vectors_file, "r", encoding="utf8") as glove:
            # fast_update, needs workaround check→delete after insert
            # val = True
            for line in tqdm(glove, total=file_len(glove_vectors_file)):
                word, vector = tuple(line.split(" ", 1))
                # vector = np.fromstring(vector, sep=" ")
                word = word_chck(word)
                # if val:
                #     # !!!!!!!!!!!!!WORD MUST BE FOUND IN DATABASE!!!!!!!!!!!
                #     check = self.find('word', word, element="rowid")
                #     if check:
                #         id = list(check)[0]
                #     else:
                #         id = None
                # if id:
                #     val = False
                #     id += 1
                e = Embedding(word, vector.replace("\n", ''))
                self.load_to_db(e, rewrite=rewrite)
            self.db.commit()
