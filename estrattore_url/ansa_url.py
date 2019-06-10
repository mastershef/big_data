import duckduckgo
urls = open("./urls_politica.txt","w+")
for link in duckduckgo.search('site:www.ansa.it/sito/notizie/politica', max_results=400):
    urls.write(link+"\n")
    print link
urls.close()
