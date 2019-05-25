import sqlite3


# with self.conn:
# Sql.select('''SELECT name FROM sqlite_master WHERE type='table''')
class Sql(object):
    def __init__(self, db_file):
        try:
            self.db_file = db_file
            self.conn = sqlite3.connect(db_file)
            self.cur = self.conn.cursor()
        except Exception as e:
            print(e.__traceback__.tb_lineno, e)

    def name(self):
        return self.db_file

    def commit(self):
        self.conn.commit()

    def execute(self, query):
        self.cur.execute(query)

    def select(self, query):
        output = self.cur.execute(query)
        rows = output.fetchall()
        # for row in rows:
        # print(row)
        return rows

    def dictk(self, entity):
        _dict = dict(entity)
        if type(_dict['id']) is not str:
            del _dict['id']
        return ", ".join('{}'.format(k) for k in _dict.keys())

    def dictv(self, entity):
        _dict = dict(entity)
        if type(_dict['id']) is not str:
            del _dict['id']
        return ", ".join('\'{}\''.format(k) for k in _dict.values())

    def dictkv(self, entity, oper=","):
        # print("hello")
        _dict = dict(entity)
        if type(_dict['id']) is not str:
            del _dict['id']
            try:
                if oper == "and":
                    del _dict['created']
                    del _dict['modified']
                    del _dict['accessed']
                    del _dict['ai']
            except:
                pass
        return (oper + " ").join('{} =\'{}\''.format(k, _dict[k]) for k in _dict.keys())

    def insert(self, table, entity):
        # ignore id
        # print(entity)
        query = "INSERT INTO " + table + "(" + self.dictk(entity) + ") VALUES " + "(" + self.dictv(entity) + ")"
        # print(query)
        self.cur.execute(query)

    def select_one(self, table, id):
        r_select = self.select("SELECT * FROM " + table + " WHERE id = '" + str(id) + "'")
        # print(r_select[0])
        if r_select:
            return r_select[0]
        else:
            return None

    def select_where(self, table, entity):
        r_select = self.select("SELECT * FROM " + table + " WHERE " + self.dictkv(entity, "and"))
        print("SELECT * FROM " + table + " WHERE " + self.dictkv(entity, " and"))
        if r_select:
            return r_select[0]
        else:
            return None

    def update(self, table, entity, id=None):
        if not id:
            id = entity.get_id()
        check = [i[0] for i in self.select("SELECT id FROM " + table)]
        if id not in check:
            query = "INSERT INTO " + table + " VALUES " + "(" + self.dictv(entity) + ")"
        else:
            query = "UPDATE " + table + " SET " + self.dictkv(entity) + " WHERE id = " + str(id)
        # print(query)
        self.cur.execute(query)

    def last_id(self, table):
        return self.select("SELECT rowid from " + table + " order by ROWID DESC limit 1 ")[0][0]

    def delete(self, table, entity):
        query = "DELETE FROM " + table + " WHERE id = " + str(entity.get_id())
        self.cur.execute(query)

    def select_one_thing(self, element, table, column, thing):
        r_select = self.select("SELECT " + element + " FROM " + table + " WHERE " + column + " = '" + str(thing) + "'")
        # print(r_select[0])
        if r_select:
            return r_select[0]
        else:
            return None

    def close(self):
        self.conn.close()
