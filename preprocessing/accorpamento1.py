import json
from os import listdir
from os.path import isfile, join

sostituzioni = open("dizionario_sostituzioni.json","r")
dizionario_sostituzioni = json.loads(sostituzioni.read())

onlyfiles = [f for f in listdir("../articoli") if isfile(join("../articoli", f))]
for json_file in onlyfiles:
    docJson = open("../articoli/"+json_file,"r")
    json_file = json_file.rjust(9,'0')
    outfile = open("../articoli_preprocessati_1/"+json_file,"w")
    jsonData = json.loads(docJson.read())
    categoria_articolo = jsonData['categoria']

    for sostituzione in dizionario_sostituzioni:
        if categoria_articolo.strip() in dizionario_sostituzioni[sostituzione]:
            jsonData['categoria'] = sostituzione
    outfile.write(json.dumps(jsonData))
    outfile.close()
    docJson.close()

sostituzioni.close()
