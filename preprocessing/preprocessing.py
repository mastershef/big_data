import json
import sys
import random
from os import listdir
from os.path import isfile, join

from nltk.tokenize import TweetTokenizer
from string import punctuation
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from string import punctuation
import re
import tqdm


def news_analyzer(news, tokenizer, stemmer, stop_words):
    # ============== YOUR CODE HERE ==============
    campi_interesse = ['testo','titolo_articolo','sottotitolo']
    for campo in campi_interesse:
        #apostrofi fix
        news[campo] = BeautifulSoup(news[campo], "lxml").get_text()
        news[campo] = news[campo].replace("'"," ")
        news[campo] = re.sub("http\S+", " link ", news[campo])
        news[campo] = tokenizer.tokenize(news[campo])
        news[campo] = [token for token in news[campo] if not token.isdigit() and token not in stop_words]
        news[campo] = [stemmer.stem(token) for token in news[campo] if not token.isdigit() and token not in stop_words]
    # ============================================

    return news

def preproc(news_dir):
    nltk.download("stopwords")
    stop_words = stopwords.words('italian') + list(punctuation)
    stemmer = SnowballStemmer("italian")
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True
    )
    X_train =  []
    onlyfiles = [f for f in listdir(news_dir) if isfile(join(news_dir, f))]
    print onlyfiles
    for articolo in onlyfiles:
        docJson = open(news_dir+articolo,"r")
        jsonData = json.loads(docJson.read())
        X_train.append(jsonData)
        docJson.close()

    return [news_analyzer(news, tokenizer, stemmer, stop_words) for news in tqdm.tqdm(X_train)]


if __name__ == "__main__":
    nltk.download("stopwords")
    stop_words = stopwords.words('italian') + list(punctuation)
    stemmer = SnowballStemmer("italian")
    tokenizer = TweetTokenizer(
        preserve_case=False,
        reduce_len=True,
        strip_handles=True
    )
    X_train =  []
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    for articolo in onlyfiles:
        docJson = open(sys.argv[1]+articolo,"r")
        jsonData = json.loads(docJson.read())
        
        X_train.append(jsonData)
        docJson.close()

    X_preproc = [news_analyzer(news, tokenizer, stemmer, stop_words) for news in tqdm.tqdm(X_train)]
    #print X_preproc[random.randint(0,len(X_preproc)-1)]

    print "categoria,"
    for i,x in enumerate(X_train):
        print x['categoria'].strip() + ","

"""
stopwords = open("stopwords.list","r")
lista_stopwords = stopwords.read().split("\n")[:-1]

onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

for articolo in onlyfiles:
    print articolo
    docJson = open(sys.argv[1]+articolo,"r")
    outfile = open(sys.argv[2]+articolo,"w")

    jsonData = json.loads(docJson.read())
    corpo_articolo = re.sub(r"([a-z])[:,\.]( )","$1$2",jsonData['testo'].replace(",","").replace(".","").replace("?","").replace("(","").replace(")","").replace("\"","").replace("'","").replace("-"," ").lower())
    titolo_articolo = re.sub(r"([a-z])[:,\.]( )","$1$2",jsonData['titolo_articolo'].replace(",","").replace(".","").replace("?","").replace("(","").replace(")","").replace("\"","").replace("'","").replace("-"," ").lower())
    sottotitolo_articolo = re.sub(r"([a-z])[:,\.]( )","$1$2",jsonData['sottotitolo'].replace(",","").replace(".","").replace("?","").replace("(","").replace(")","").replace("\"","").replace("'","").replace("-"," ").lower())


    corpo_articolo = " ".join([word for word in corpo_articolo.split(" ") if word not in lista_stopwords])
    titolo_articolo = " ".join([word for word in titolo_articolo.split(" ") if word not in lista_stopwords])
    sottotitolo_articolo = " ".join([word for word in sottotitolo_articolo.split(" ") if word not in lista_stopwords])

    jsonData['testo'] = corpo_articolo
    jsonData['titolo_articolo'] = titolo_articolo
    jsonData['sottotitolo'] = sottotitolo_articolo

    outfile.write(json.dumps(jsonData))

    docJson.close()
    outfile.close()
stopwords.close()
print lista_stopwords
"""
