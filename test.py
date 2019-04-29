# import re
# from entity.patternrepo import PatternDbRepo
# from entity.pattern import Pattern
#
#
# mdb = PatternDbRepo()
# # m = Message("hello", "hello world", 1)
#
# print(mdb.save(Pattern("", "hello|hi|good morning", 1)))
#
# # print(mdb.get(27))
from entity.conversationrepo import ConversationDbRepo

cdb = ConversationDbRepo()
m = [1,2,3,4,5]
print(m[1:-3])
