import os
# print([file[:-3] for file in os.listdir(".") if not file.startswith("__") and file.endswith(".py")])
# print("entity")
# print(os.getcwd())

__all__ = [file[:-3] for file in os.listdir(".") if not file.startswith("__") and file.endswith(".py")]
# __all__=['category', 'categoryrepo', 'conversation', 'conversationrepo', 'databaserepo', 'embedding', 'embeddingrepo', 'entity', 'factoryrepo', 'message', 'messagerepo', 'pattern', 'patternrepo', 'policy', 'policyrepo', 'user', 'userrepo']
# print(__all__)

