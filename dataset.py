import pandas as pd
import numpy as np
from sklearn.utils import shuffle



df = pd.read_csv('./sent140.csv',encoding = "ISO-8859-1")
df.columns = ['sentiment','id','date','query','user','tweet']

print(df.head())

#print(df['Sentiment'].count())	

print(df.shape)

df = df[['sentiment','tweet']]

#print(df.sentiment.value_counts()[0])

df = shuffle(df)

print(df.head())

df.to_csv('tweet_train.csv',encoding='utf-8',index=False)
