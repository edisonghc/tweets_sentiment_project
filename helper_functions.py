"""
Authors: Xinyue Li

    Additional helper functions for the program
    Append any function here to `main_classical.py` before using

"""

from FeatureExtractor import *
from LogisticRegression import *
from Evaluation import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
from  nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

# dataset information
TRAIN_FILEPATH = "./example_training/input/training.1600000.processed.noemoticon.csv"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# text cleaning
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



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


def run_on_BoW_LR():
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




