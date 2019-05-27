# import logging
#
# from entity.message import Message
# from entity.messagerepo import MessageDbRepo
# from entity.userrepo import UserDbRepo
# from entity.userrepo import PatternDbRepo
#
# logging.basicConfig(level=logging.DEBUG)
# from datetime import datetime
#
# time = int(datetime.timestamp(datetime.now()))
# # udb = UserDbRepo()
# # udb.save(User("Ab", "bc", "hello@com", "2302303013", time, time, time))
#
# # m = Message("hello", "hello world", 1)
# mdb = MessageDbRepo()
# m = Message("asdfdffl", "hello world", 1)
# # m.id = 17
# # print(mdb.save(m))
# print(mdb.get(27))
# from entity.patternrepo import PatternDbRepo
#

from entity.embedding import Embedding
from entity.embeddingrepo import EmbeddingDbRepo
from entity.patternrepo import PatternDbRepo


def upload_embeddings(glove_vectors_file, rewrite):
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

    emrep = EmbeddingDbRepo()
    with open(glove_vectors_file, "r", encoding="utf8") as glove:
        for line in tqdm(glove, total=file_len(glove_vectors_file)):
            word, vector = tuple(line.split(" ", 1))
            # vector = np.fromstring(vector, sep=" ")
            word = word_chck(word)
            e = Embedding(word, vector.replace("\n", ''))
            emrep.load_to_db(e, rewrite=rewrite)
        emrep.save_db()


# file = 'glove.6B.50d.txt'
# path = './neural/'
# file_path = path + file
# emrep = EmbeddingDbRepo()
#
# upload_embeddings(file_path, False)
# print(emrep.find('word', 'life'))


# from entity.embeddingrepo import EmbeddingDbRepo
# import numpy as np
#
# glove_wordmap = {}
# embrepo = EmbeddingDbRepo()
# gloves = embrepo.get()
# for glove in gloves:
#     name, vector = list(glove)[1], list(glove)[2]
#     glove_wordmap[name] = np.fromstring(vector, sep=" ")

# file = 'glove.6B.50d.txt'
# path = './neural/'
# file_path = path + file
# emrep = EmbeddingDbRepo()
# emrep.upload_embeddings(file_path)
from entity.patternrepo import PatternDbRepo
pdb = PatternDbRepo()
path = "./neural/tasks_1-20_v1-2/en/"
file = "qa5.txt"
file_path = path + file
print(file_path)
pdb.upload_qa_to_db(file_path)
