import pandas as pd
import numpy as np
import os
import re
import html
import pyprind
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle




stopwords = stopwords.words('english')

def tokenizer(text):
	text = html.unescape(text)
	text = re.sub('http://[a-zA-Z0-9./]+','',text) 
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	tokenized = [w for w in text.split() if w not in stopwords]
	return tokenized


def data_stream(path):
	
	with open(path,'r') as csv:
		next(csv)
		for line in csv :
			text ,label = line[2:],int(line[0])
			yield text,label    

# print(next(stream_docs('./tweet_train.csv')))

def mini_batch(doc_stream,size):

	docs,y = [],[]

	try:

		for _ in range(size):

			text,label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None, None
	return docs,y



vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
						preprocessor=None,tokenizer=tokenizer)

clf = SGDClassifier(loss='log',random_state=1,n_iter=1)

pbar=pyprind.ProgBar(1500)
classes = np.array([0,4])


doc_stream = data_stream('./tweet_train.csv')

for _ in range(1500):
	X_train,y_train = mini_batch(doc_stream,size = 1000)

	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes)
	pbar.update()

dest = './pkl_objects'
if not os.path.exists(dest):
	os.makedirs(dest)
pickle.dump(clf,open(os.path.join(dest, 'tweet140_clf.pkl'), 'wb'),protocol=4)

X_test, y_test = mini_batch(doc_stream,size =5000)
#print(X_test[:5])
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

#clf = pickle.load(open(os.path.join('pkl_objects', 'tweet140_clf.pkl'), 'rb')) 	


def classify(text):

	px=vect.transform([text])
	labels = {'[4]':'Positive' , '[0]':'Negative'}
	print(labels[str(clf.predict(px))])
	print(clf.predict_proba(px))	
	return labels[str(clf.predict(px))]














