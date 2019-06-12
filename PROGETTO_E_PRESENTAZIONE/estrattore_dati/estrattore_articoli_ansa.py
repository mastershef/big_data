import requests
import logging
from lxml import html
import re
import json
import os
import sys


#definisco headers
headers = {
    "User-Agent":"Google Chrome Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'",
}

"""
Funzione che estrae titolo, sottotitolo, cateogira, tag
per un url di articolo ansa

Parametri:
    - URL: link articolo ansa
Ritorna:
    Dizioario con: {"titolo_articolo":...,"sottotitolo":....,"testo":...,tags:["tag1","tag2",...],"categoria:....}
"""
def estrai(url):
    session = requests.Session() #ottengo sessione di requests
    session.headers.update(headers) #imposto i header
    logging.info("Scarico pagina "+url)
    page = session.get(url)
    tree = html.fromstring(page.content) #creo DOM della pagina

    try:
        title = tree.xpath("//*[@itemprop='headline']")[0].text_content()  #via xpath per il titolo
    except:
        title = ""
    try:
        subtitle = tree.xpath("//*[@class='news-stit']")[0].text_content() #via xpath per il sottotitolo
    except:
        subtitle = ""
    try:
        testo_articolo = tree.xpath("//div[@itemprop='articleBody']")[0].text_content().strip() #xpath per il testo dell'articolo
    except:
        testo_articolo = ""

    tags_javascript = tree.xpath("//div[@class='tag-news']/div/script") #divisore con i tag


    try:
        tipo  =  tree.xpath('//meta[@property="og:type"]')[0].get('content')
        categoria = tree.xpath("//meta[@itemprop='articleSection']")[0].get('content')
    except:
        categoria = "NULL"

    tagEstratti = []
    for tags in tags_javascript:
        tags_con_virgola = tags.text_content().replace('displayTags("','').replace(');','').replace('"','').strip()
        tagEstratti=tagEstratti+ [tag.lower().strip() for tag in  tags_con_virgola.split(",")]
    #tags = div_tags.xpath(".//div/ul/li") #cerco dentro div_tags
    return {"categoria":categoria,"titolo_articolo":title, "sottotitolo": subtitle, "testo": testo_articolo, "tags":tagEstratti}



if __name__ == "__main__":
    cat = sys.argv[1]
    #leggi il file con gli articoli
    articoli = open("../estrattore_url/urls_"+cat+".txt","r")
    i = 1

    for articolo in articoli:
        #print articolo
        exists = os.path.isfile("../articoli_"+cat+"/"+str(i)+".json")
        if not exists:
            art = estrai(articolo.strip());
            if art != "not_article":
                f = open("../articoli_"+cat+"/"+str(i)+".json","w+")
                json.dump(art, f)
                f.close()
                i=i+1
