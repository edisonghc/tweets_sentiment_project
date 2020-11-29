#!/usr/bin/env python
# coding: utf-8

# In[1]:
from Evaluation import print_results
from FeatureExtractor import CountVectorizer
from LogisticRegression import LogisticRegression
from Evaluation import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
import time
from collections import Counter
import re
from importlib import reload

import nltk

from nltk.tokenize import word_tokenize
from  nltk.stem import SnowballStemmer

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# dataset information
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# text cleaning
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

train_filepath = "./example_training/input/training.1600000.processed.noemoticon.csv"

#data split parameter
SPLIT_SIZE = 0.25

# In[2]:
# open and read data file
print("Open file:", "training_data.csv")
df = pd.read_csv(train_filepath, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# print the size of the data set
print("Dataset size:", len(df))
# shuffle the data set
df = df.sample(frac=0.1).reset_index(drop=True)

# In[3]:
# preprocess method
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

print(">>>>>>>>>>Preprocessing")
# apply preprocess method
df.text = df.text.apply(lambda x: preprocess(x))
print("DONE!")


# In[4]
# building vocabulary

# print(">>>>>>>>>>Tokenizing")
# nltk.download('punkt')
# df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
# print("DONE!")
# df = df[:10000]

#df = df.sample(frac=0.1).reset_index(drop=True)
tweets_for_training, tweets_for_testing, y_tr, y_te = train_test_split(df, df['text'], test_size=SPLIT_SIZE, random_state=0)

print(">>>>>>>>>>Building Vocabulary")
# feature_collection = Features()
# for id in tweets_for_training.index:
#     feature_collection.add_to_vocab(df['text'][id])

tweet_train = np.array(tweets_for_training['text'])
y_train = np.array(tweets_for_training['target']/4, dtype=int)

tweet_test = np.array(tweets_for_testing['text'])
y_test = np.array(tweets_for_testing['target']/4, dtype=int)

#print(tweet_train)
cv = CountVectorizer()
# X_train = np.array(cv.word2vec_init(tweet_train))
# X_test = np.array(cv.word2vec(tweet_test))
X_train = cv.fit_transform(tweet_train)
X_test = cv.transform(tweet_test)

print("DONE!")
print("Vocabulary size: ", X_train.shape[1])
#print(feature_collection.train_vocab)

# In[5]
print(">>>>>>>>>>Training model")
# model = train_LR(tweets_for_training, feature_collection, feature_collection.train_vocab)

model = LogisticRegression()
history = model.fit(X_train, y_train)
print("DONE!")

plt.plot(history)
plt.show()

# In[6]
print(">>>>>>>>>>Classifying")
predictions = []

prediction = model.predict(X_test)

# for id in tweets_for_testing.index:
#     predict = model.predict(df['text'][id], feature_collection)
#     predictions.append(predict)
print("DONE!")

print(">>>>>>>>>>Evaluating")
# targets = tweets_for_testing['target']
# modified_targets = targets.replace(4, 1)
# evaluation(modified_targets.values.tolist(), predictions)
accuracy = np.mean(y_test == prediction)
print(f'Accuracy is {accuracy*100:.4f}%')

#Print out Confsuion matrix and Classification Report
print_results(prediction,y_test)
print("DONE!")
