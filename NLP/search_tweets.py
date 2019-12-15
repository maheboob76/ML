#https://www.codementor.io/ferrorodolfo/sentiment-analysis-on-trump-s-tweets-using-python-pltbvb4xr

# -*- coding: utf-8 -*-
"""Created on Wed Nov 27 11:42:18 2019

@author: Amaan
"""
# For plotting and visualization:
from IPython.display import display
# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing


import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd



# We import our access keys:
from credentials import *    # This will allow us to use the keys as variables

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


# We create an extractor object:
api = twitter_setup()

# We create a tweet list as follows:
#tweets = api.user_timeline(screen_name="realDonaldTrump", count=1)
tweets = api.search(q='#CABPolitics -filter:retweets', result_type='popular  ', lang='en', count=10)

print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("recent tweets:\n")
for tweet in tweets[:15]:
    print(tweet.text)
    print()

csvFile = open('CABPolitics_RT2.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#CABPolitics -filter:retweets",count=100,
                           lang="en",
                           since="2019-12-14").items(500):
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')]) #encode is required else it fails