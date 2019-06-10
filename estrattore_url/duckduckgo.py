import requests
from lxml import html
import time

def search(keywords, max_results=None):
	url = 'https://duckduckgo.com/html/'
	headers = {
		"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0"
	}

	params = {
		'q': keywords,
		'b': '',
		'kl': 'us-en'
	}

	yielded = 0
	while True:
		res = requests.post(url, data=params,headers=headers)
		doc = html.fromstring(res.text)
		print res
		results = [a.get('href') for a in doc.cssselect('.result__snippet')]
		for result in results:
			yield result
			time.sleep(0.1)
			yielded += 1
			if max_results and yielded >= max_results:
				return

		try:
			form = doc.cssselect('.nav-link form')[0]
			print form
		except IndexError:
			return
		params = dict(form.fields)
		print params
