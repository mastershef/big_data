from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from os import listdir
from scipy.sparse import find, csr_matrix
import numpy
import nltk
from string import punctuation
import re
import tqdm
from nltk.corpus import stopwords
import pandas as pd
from os.path import isfile, join
import sys
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import json
from bs4 import BeautifulSoup


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
    for articolo in onlyfiles:
        docJson = open(news_dir+articolo,"r")
        jsonData = json.loads(docJson.read())

        X_train.append(jsonData)
        docJson.close()

    return [news_analyzer(news, tokenizer, stemmer, stop_words) for news in tqdm.tqdm(X_train)]



def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

documents = [" ".join(x['testo']) for x in preproc(sys.argv[1])]
#print documents
no_features = 5000
tf_vectorizer = CountVectorizer(max_df=0.05,min_df=6, encoding='utf-8', binary =True,max_features=no_features, lowercase=False)
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
#print(tf_vectorizer.get_feature_names())
# tdf = numpy.sum(tf.toarray(), axis=0)
# df ={}
# for i in tdf:
#     if i in df:
#         df[i] += 1
#     else:
#         df[i] = 1
#
# k = df.keys()
# for item in k:
#     print str(item) + ","+str(df[item])

#print "Feature length: " + str(len(tf_vectorizer.get_feature_names()))
no_topics = 7
# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=50, learning_method='online', learning_offset=60.,random_state=0).fit(tf)


no_top_words = 20
display_topics(lda, tf_feature_names, no_top_words)

# Create Document - Topic Matrix
lda_output = lda.transform(tf)

# column names
topicnames = ["Topic" + str(i) for i in range(no_topics)]

# index names
docnames = ["Doc" + str(i) for i in range(len(documents))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(numpy.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = numpy.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15)
print df_document_topics
