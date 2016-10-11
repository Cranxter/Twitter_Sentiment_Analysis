import tweepy 
import json
import classifier
import sys


# consumer_key=""
# consumer_secret=""
# access_token=""
# access_secret=""

tag = sys.argv[1]

class StreamListener(tweepy.StreamListener):

	def on_data(self,data):
		
		data = json.loads(data)
		try:
			tweet_text = data["text"]		
			value = classifier.classify(tweet_text)

			out = open('sentiment_output.txt',"a")
			out.write(value)
			out.write("\n")
			out.close()

			print(tweet_text)
			#print(value)
			print("\n")
			return True
		except:
			print("blank")	

	def on_status(self,status):
		print(status)	

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)		

api = tweepy.API(auth)

open("sentiment_output.txt", 'w').close()
myStreamListener = StreamListener()
stream = tweepy.Stream(auth, StreamListener())
stream.filter(languages=["en"],track = [tag])
