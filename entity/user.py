from .entity import Entity


class User(Entity):
    def __init__(self, first_name, last_name, email, phone, id=None, ai=False):
        self.id = id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = phone
        self.ai = bool(ai)
        super().__init__()

    def __delete__(self, id):
        return self.id

    def get_first_name(self):
        return self.first_name

    def get_last_name(self):
        return self.last_name

    def get_name_full(self):
        return self.get_first_name() + self.get_last_name()

    def get_email(self):
        return self.email

    def get_phone(self):
        return self.phone
