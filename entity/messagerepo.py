from entity.message import Message
from .databaserepo import DatabaseRepo


class MessageDbRepo(DatabaseRepo):
    def __init__(self):
        super().__init__("Messages")

    def all(self, cid):
        query = "SELECT * FROM " + self.table + " WHERE conversation_id = '" + str(cid) + "'"
        m_results = self.db.select(query)
        if m_results:
            m = [Message(*m_res[1:-3], m_res[0]) for m_res in m_results]
            return m
        else:
            return None

    def get(self, id):
        current = super().get(id)
        m = Message(*current[:-3], id)
        m.created, m.modified, m.accessed = current[3], current[4], current[5]
        return m

    def last(self, cid):
        query = "SELECT * FROM " + self.table + " WHERE conversation_id = '" + str(
            cid) + "' ORDER BY created DESC LIMIT 0,1"
        m_results = self.db.select(query)
        if m_results:
            m_results = m_results[0]
            m = Message(*m_results[1:-3], m_results[0])
            return m
        else:
            return None
