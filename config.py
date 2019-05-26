import json
import os
print(os.getcwd())
def get_config(key):
    openfile = open("config.json", 'r', encoding='utf-8')
    test = json.loads(openfile.read())
    return test[key]

