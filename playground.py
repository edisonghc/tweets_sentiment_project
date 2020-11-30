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

tweets_for_training, tweets_for_testing, y_tr, y_te = train_test_split(df, df['text'], test_size=SPLIT_SIZE, random_state=0)

print(">>>>>>>>>>Building Vocabulary")


tweet_train = np.array(tweets_for_training['text'])
y_train = np.array(tweets_for_training['target']/4, dtype=int)

tweet_test = np.array(tweets_for_testing['text'])
y_test = np.array(tweets_for_testing['target']/4, dtype=int)


cv = CountVectorizer()
# X_train = np.array(cv.word2vec_init(tweet_train))
# X_test = np.array(cv.word2vec(tweet_test))
X_train = cv.fit_transform(tweet_train)
X_test = cv.transform(tweet_test)

print("DONE!")
print("Vocabulary size: ", X_train.shape[1])


# In[5]
print(">>>>>>>>>>Training model")

model = LogisticRegression()
history = model.fit(X_train, y_train)
print("DONE!")

plt.plot(history)
plt.show()

# In[6]
print(">>>>>>>>>>Classifying")
predictions = []

prediction = model.predict(X_test)


print("DONE!")

print(">>>>>>>>>>Evaluating")

accuracy = np.mean(y_test == prediction)
print(f'Accuracy is {accuracy*100:.4f}%')

#Print out Confsuion matrix and Classification Report
print_results(prediction,y_test)
print("DONE!")



def write_output(weights, output_path):
    """
    Author: Xinyue Li
    write the weights data into the file, one element each line
    :param weights: an array of weights
    :param output_path: the path of the output file
    """
    #open the output file
    write_file = open(output_path, 'w')

    #write the data into the file
    for l in weights:
        write_file.write(str(l) + "\n")
    
    #close the file 
    write_file.close()


def run_on_BoW_LR:
    """
    Author: Xinyue Li
    Train, test and evalute with the Bag-of-Words option and Logistic Regression model
    """
    print(">>>>>>>>>>Preprocessing")
    # apply preprocess method
    df.text = df.text.apply(lambda x: preprocess(x))
    print("DONE!")

    #tokenize
    print(">>>>>>>>>>Tokenizing")
    nltk.download('punkt')
    df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
    print("DONE!")

    #shuffle the data set and choose the training part and testing part
    df = df.sample(frac=1).reset_index(drop=True)
    tweets_for_training = df.head(1000000)
    tweets_for_testing = df.tail(1000)

    #build up the vocabulary 
    print(">>>>>>>>>>Building Vocabulary")
    feature_collection = Features()
    for id in tweets_for_training.index: 
        feature_collection.add_to_vocab(df['text'][id])
    print("DONE!")
    print("Vocabulary size: ", len(feature_collection.train_vocab))

    #train the model 
    print(">>>>>>>>>>Training model")
    model = Logistic_Regression_for_BoW(tweets_for_training, feature_collection, feature_collection.train_vocab)
    print("DONE!")

    #write the weights and features to the output files
    write_output(model.weights, "output5.txt")
    write_output(feature_collection.train_vocab, "features5.txt")

    #classify
    print(">>>>>>>>>>Classifying")
    predictions = []
    for id in tweets_for_testing.index: 
        predict = model.predict(df['text'][id], feature_collection)
        predictions.append(predict)
    print("DONE!")

    #evaluate the accuracy 
    print(">>>>>>>>>>Evaluating")
    targets = tweets_for_testing['target']
    modified_targets = targets.replace(4, 1) # 4 is positive in the data file
    evaluation(modified_targets.values.tolist(), predictions)
    print("DONE!")

#run_on_BoW_LR

def test_on_BoW_LR:
    """
    Author: Xinyue Li
    Make the prediction on the testing data set basd on the trained LR model
    """
    # open and read data file
    print("Open file:", "ExtractedTweets.csv")
    df = pd.read_csv("ExtractedTweets.csv", encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
    
    #neglect the first row and shuffle the data
    df = df.iloc[1:]
    df = df.sample(frac=1).reset_index(drop=True)
    df_democrat = df.loc[df['party'] == "Democrat"]
    df_republican = df.loc[df['party'] == "Republican"]
    
    #print the size of the data set
    print("Dataset size:", len(df))
    print("Democrat dataset size", len(df_democrat))
    print("Republican dataset size", len(df_republican))
    
    print(">>>>>>>>>>Preprocessing")
    #apply preprocess method
    df.tweet = df.tweet.apply(lambda x: preprocess(x))
    df_democrat.tweet = df_democrat.tweet.apply(lambda x: preprocess(x))
    df_republican.tweet = df_republican.tweet.apply(lambda x: preprocess(x))
    print("DONE!")

    #tokenize
    print(">>>>>>>>>>Tokenizing")
    nltk.download('punkt')
    df['tweet'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
    print("DONE!")

    #read weights and features from the file
    weights = np.genfromtxt("output5.txt", delimiter=' ')
    features = []
    with open("features5.txt") as file_in:
        for feature in file_in:
            features.append(feature)

    #set up the LR model
    model = Logistic_Regression()
    model.weights = weights
    feature_extract = Features()
    feature_extract.train_vocab = features

    #classify and count results on democrats
    print(">>>>>>>>>>Classifying")
    predictions_d = []
    for id in df_democrat.index: 
        predict = model.predict(df_democrat['tweet'][id], feature_extract)
        predictions_d.append(predict)
    pos_democrat = predictions_d.count(1)
    neg_democrat = predictions_d.count(0)

    #classify and count results on republicans
    predictions_r = []
    for id in df_republican.index: 
        predict = model.predict(df_republican['tweet'][id], feature_extract)
        predictions_r.append(predict)
    pos_republican = predictions_r.count(1)
    neg_republican = predictions_r.count(0)

    #print out the prediction 
    print("DONE!")
    print("Democrat : ",predictions_d)
    print("Democrat positive: ",pos_democrat)
    print("Democrat negative: ",neg_democrat)

    print("Republican: ", predictions_r)
    print("Republican positive: ",pos_republican)
    print("Republican negative: ",neg_republican)

#test_on_BoW_LR