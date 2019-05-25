import os

print([file[:-3] for file in os.listdir(".") if not file.startswith("__") and file.endswith(".py")])
__all__=[file[:-3] for file in os.listdir(".") if not file.startswith("__") and file.endswith(".py")]
print("sql")
print(os.getcwd())
print(__all__)