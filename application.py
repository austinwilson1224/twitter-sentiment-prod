# from math import e
from flask import Flask, render_template, request, redirect, url_for
import tweepy
from keys import consumer_key, consumer_secret, access_token, access_token_secret
# from model import data #setup, apply_prediction


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd
import tensorflow as tf

import os

print(os.getcwd())
test = str(os.getcwd())
# data = pd.read_csv('twitter_training.csv', names=[
#                    "Tweet_ID", "Entity", "Sentiment", "Text"])
# data = data[['Text', 'Sentiment']]
# data = data[data.Sentiment != "Neutral"]
# data = data[data.Sentiment != "Irrelevant"]
# data.Text = data.Text.apply(lambda x: str(x).lower())
# data.Text = data.Text.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# lemmatiser = WordNetLemmatizer()
# stopwords = set(stopwords.words())


# def remove_stopwords(ls):
#     ls = [lemmatiser.lemmatize(word) for word in ls if word not in (
#         stopwords) and (word.isalpha())]
#     ls = " ".join(ls)
#     return ls


# data.Text = data.Text.apply(word_tokenize)
# data.Text = data.Text.apply(remove_stopwords)

# for idx, row in data.iterrows():
#     row[0] = row[0].replace('rt', ' ')

# max_features = 1000
# tokenizer = Tokenizer(num_words=max_features, split=' ')
# tokenizer.fit_on_texts(data.Text.values)
# X = tokenizer.texts_to_sequences(data.Text.values)
# X = pad_sequences(X)

# embed_dim = 128
# lstm_out = 196

# print(X)
# print(consumer_key, consumer_secret, access_token, access_token_secret)
# authorization


# try:
#     auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_token, access_token_secret)
#     api = tweepy.API(auth)
#     me = api.me()
#     name = me.screen_name
#     tweets = api.user_timeline(name)
#     setup()
# except:
#     me = None
#     name = None
#     tweets = None
#     print('test')

tweets = [
    {
        "id": 1,
        "text": 'somestuff'
    },
    {
        "id": 2,
        "text": 'someotherstuff'
    }
]


application = Flask(__name__)


@application.route("/", methods=["GET", "POST"])
def home():
    data = pd.read_csv(url_for('static', filename='models/twitter_training.csv'), names=[
                   "Tweet_ID", "Entity", "Sentiment", "Text"])
    print(data.iloc[0])
    # return "<h1>Hello</h1>"
    # if request.method == "POST":
    #     content = request.form['content']
    #     new_status = api.update_status(content)
    #     print(content)
    tweet_data = []
    # try:
    #     tweets = api.user_timeline(name)
        
    # except:
    #     tweets = None
    sentiments = []
    tweets = [
        {
            "id": 1,
            "text": 'somestuff'
        },
        {
            "id": 2,
            "text": 'someotherstuff'
        }
    ]
    name=None
    # try:
    #     for tweet in tweets:
    #         sentiment = apply_prediction(tweet.text)
    #         sentiments.append(sentiment)
    #         tweet_data.append({
    #             "text": tweet.text,
    #             "id": tweet.id,
    #             "sentiment": sentiment
    #         })
    #     #print(tweet_data)
    #     tweets = tweet_data
    #     #print(tweets)
    # except:
    #     print("something with apply prediction fxn")

    return test#str(X[0])#render_template("index.html", name=name, tweets=tweets)


# @application.route("/tweet:<id>", methods=["GET", "POST"])
# def get_single_tweet(id):

#     sentiment = None
#     try:
#         #print(id)
#         tweet = [api.get_status(id)]
#         #print(tweet)
#         tweet = [{
#             "text": tweet[0].text,
#             "id": id,
#             "sentiment": apply_prediction(tweet[0].text)
#         }]
#         print(tweet)

#     except:
#         tweet = None
#         print('error getting tweet!!!!')

#     # sentiment = apply_prediction(tweet[0].text)
#     return render_template("index.html", tweets=tweet)


# @application.route("/status", methods=["GET", "POST"])
# def post_status():
#     return render_template("post_status.html", name=name)


# @application.route("/remove_status:<id>", methods=["GET", "POST"])
# def get_status(id):
#     print(id)
#     try:
#         api.destroy_status(id)
#     except:
#         print('error')
#     return redirect("/")


if __name__ == "__main__":
    application.run(debug=False)
