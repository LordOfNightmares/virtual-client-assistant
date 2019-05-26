import os

from unidecode import unidecode

from entity.embeddingrepo import EmbeddingDbRepo
from sql.sqldb import Sql

db_install = [file for file in os.listdir('sql') if file.endswith(".sql")]

from subprocess import Popen, PIPE, STDOUT


def install_db(db=Sql('server.db')):
    con_db = db
    for query in db_install:
        openfile = open(os.path.abspath("sql\\" + query), 'r', encoding='utf-8')
        read_line = [unidecode(line.strip()) for line in openfile]
        read_line = '\n'.join(read_line)
        con_db.execute(read_line)
    con_db.commit()
    con_db.close()


# try:
# import os
# import stat
#
# file_name = 'server.db'
# try:
#     os.remove(file_name)
# except PermissionError:
#     print('PermissionError do change')
#     os.chmod(file_name, stat.S_IWRITE)
#     os.remove(file_name)
#
# from time import sleep
# sleep(1)
# from subprocess import check_output
# check_output("DEL /F /Q /A server.db")
install_db()

from entity.patternrepo import PatternDbRepo
pdb = PatternDbRepo()
path = "./neural/tasks_1-20_v1-2/en/"
file = "qa5.txt"
file_path = path + file
print(file_path)
pdb.upload_qa_to_db(file_path)

# pdb.all(8)
# pdb.db.close()

file = 'glove.6B.50d.txt'
path = './neural/'
file_path = path + file
emrep = EmbeddingDbRepo()
emrep.upload_embeddings(file_path, False)
# print(emrep.find('word', 'life'))
