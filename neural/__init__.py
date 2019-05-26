import os
# print(os.listdir())
__all__ = [file[:-3] for file in os.listdir("./neural/") if not file.startswith("__") and file.endswith(".py")]
print("neural")
print(os.getcwd())
print(__all__)
