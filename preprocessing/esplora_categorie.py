import json
from os import listdir
from os.path import isfile, join
import sys

onlyfiles = [f for f in listdir("../articoli") if isfile(join("../articoli", f))]

categorie = []
for json_file in onlyfiles:
    docJson = open(sys.argv[1]+json_file)
    jsonData = json.loads(docJson.read())
    categorie.append(jsonData['categoria'])
    docJson.close()

print set(categorie)
