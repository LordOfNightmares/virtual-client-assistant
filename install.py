import os

from unidecode import unidecode

from sql.sqldb import Sql

db_install = [file for file in os.listdir('sql') if file.endswith(".sql")]

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
install_db()
# except sqlite3.Error as de:
#     print(de.__traceback__.tb_lineno, de)
