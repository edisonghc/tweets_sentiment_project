#!/usr/bin/env python
# coding: utf-8

# In[18]:

from FeatureExtractor import *
from LogisticRegression import *
from Evaluation import *
import numpy as np
import pandas as pd

import os
import time
from collections import Counter
import re

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from  nltk.stem import SnowballStemmer
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# dataset information
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# text cleaning
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


# open and read data file
print("Open file:", "training_data.csv")
df = pd.read_csv("training_data.csv", encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# print the size of the data set
print("Dataset size:", len(df))

"""
# decode sentiment
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]


# store targets
df.target = df.target.apply(lambda x: decode_sentiment(x))


#count targets
target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")
plt.show()
"""

"""
print("before processing: ", df.iloc[0].text)
print("label: ", df.iloc[0].target)
"""


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

"""
print("after processing: ", df.iloc[0].text)
print("label: ", df.iloc[0].target)
"""

#tokenize
print(">>>>>>>>>>Tokenizing")
nltk.download('punkt')
df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
print("DONE!")

df = df.sample(frac=1).reset_index(drop=True)
tweets_for_training = df.head(1000)
tweets_for_testing = df.tail(100)

print(">>>>>>>>>>Building Vocabulary")
feature_collection = Features()
for id in tweets_for_training.index: 
    feature_collection.add_to_vocab(df['text'][id])
print("DONE!")
print("Vocabulary size: ", len(feature_collection.train_vocab))
#print(feature_collection.train_vocab)

"""
print(">>>>>>>>>>Extracting Feature")
current_features = []
for id in df.tail(1).index:
    current_features = feature_collection.extractor(df['text'][id], feature_collection.train_vocab)
print("DONE!")
print("Vocabulary size: ", len(feature_collection.train_vocab))
print(current_features)
"""

print(">>>>>>>>>>Training model")
model = train_LR(tweets_for_training, feature_collection, feature_collection.train_vocab)
print("DONE!")

print(">>>>>>>>>>Classifying")
predictions = []
for id in tweets_for_testing.index: 
    predict = model.predict(df['text'][id], feature_collection)
    predictions.append(predict)
print("DONE!")

print(">>>>>>>>>>Evaluating")
targets = tweets_for_testing['target']
modified_targets = targets.replace(4, 1)
evaluation(modified_targets.values.tolist(), predictions)
print("DONE!")






    



