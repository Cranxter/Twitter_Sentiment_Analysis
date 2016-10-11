# Twitter_Sentiment_Analysis
Live Classification of a Twitter stream using Out-of-Core Learning.

##Synopsis

<p>This project demonstrates how out of core learning can be used to train on a very large dataset (~1.6 million entries) in a relatively short span of time. Tweets are continuously streamed and the predicted sentiments are plotted live on a graph.</p>
<p>The classification is done by a Logistic Regression Classifier which iterates over the dataset in batches of 1000 documents and utilises the partial_fit function.<br>
Since all feature vectors will not be in memory at a given time, a Hashing Vecotrizer is used instead of a Tfidf Vectorizer.</p>

For a more detailed discussion refer http://scikit-learn.org/stable/modules/scaling_strategies.html 


## Prerequisites
1. scikit-learn
2. NLTK
3. Tweepy
4. Matplotlib
5. pyprind


## Note

* The dataset is not present in the repo owing to its large size.
You can download the dataset from [Sentiment140](http://help.sentiment140.com/for-students/)
* Please remember to enter your own twitter developer credentials in the twitter_Stream.py file

```python
consumer_key=""
consumer_secret=""
access_token=""
access_secret=""
```


## External Links

* http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
