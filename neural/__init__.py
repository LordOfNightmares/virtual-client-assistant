import os

__all__ = [file[:-3] for file in os.listdir("./entity") if not file.startswith("__") and file.endswith(".py")]
print("neural")
print(os.getcwd())
print(__all__)
