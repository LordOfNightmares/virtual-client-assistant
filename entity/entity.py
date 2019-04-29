from datetime import datetime


class Entity(object):
    def __init__(self, created=0, modified=0, accessed=0):
        self.created = created
        self.modified = modified
        self.accessed = accessed

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __str__(self):
        _dict = dict(self)
        return "{" + ",\n ".join('{}:\'{}\''.format(k, _dict[k]) for k in _dict.keys()) + "}"

    def get_id(self):
        return self.id

    def now(self):
        return int(datetime.timestamp(datetime.now()))

    def get_created(self):
        return self.created

    def get_modified(self):
        return self.modified

    def get_accessed(self):
        return self.accessed


def t_print(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%H:%M %d-%m-%Y")
