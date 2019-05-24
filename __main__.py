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
from entity.patternrepo import PatternDbRepo

pdb = PatternDbRepo()
path = "./neural/tasks_1-20_v1-2/en/"
file = "qa5_three-arg-relations_test.txt"
file_path = path + file
# pdb.read_dataset(file_path)
pdb.all(8)