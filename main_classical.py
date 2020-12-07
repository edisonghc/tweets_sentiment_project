"""
Authors: Edison Gu, Xinyue Li, Simon Manning

    Explore the model performance using Bag-of-Word and Word2Vec embedding,
    using Logistic Regression
    
"""

from FeatureExtractor import CountVectorizer
from LogisticRegression import LogisticRegression
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

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# dataset information
TRAIN_FILEPATH = "./example_training/input/training.1600000.processed.noemoticon.csv"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# text cleaning
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# Train-Test Split and Sampling parameters
SPLIT_SIZE = 0.25
SAMPLE_FRAC = 0.01

# open and read data file
print("Open file:", "training_data.csv")
df = pd.read_csv(TRAIN_FILEPATH, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
print("Dataset size:", len(df))

# shuffle the data set
df = df.sample(frac=SAMPLE_FRAC).reset_index(drop=True)

# preprocess method
def preprocess(text, stem=False):
    """
    Author: Edison Gu
    """
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

# Preprocessing
print(">>>>>>>>>>Preprocessing")

df.text = df.text.apply(lambda x: preprocess(x))

print("DONE!")

# Train-Test Split
tweets_for_training, tweets_for_testing, y_tr, y_te = train_test_split(df, df['text'], test_size=SPLIT_SIZE, random_state=0)

tweet_train = np.array(tweets_for_training['text'])
y_train = np.array(tweets_for_training['target']/4, dtype=int)

tweet_test = np.array(tweets_for_testing['text'])
y_test = np.array(tweets_for_testing['target']/4, dtype=int)

# Build Vocabulary
print(">>>>>>>>>>Building Vocabulary")

cv = CountVectorizer()

# Uncomment one of the below for Bag-of-Word or Word2Vec embedding

# X_train = cv.fit_transform(tweet_train)
# X_test = cv.transform(tweet_test)

X_train = np.array(cv.word2vec_init(tweet_train))
X_test = np.array(cv.word2vec(tweet_test))

print("DONE!")

print("Training shape: ", X_train.shape)
print("Testing shape: ", X_test.shape)

# Train Model
print(">>>>>>>>>>Training model")

model = LogisticRegression()

history = model.fit(X_train, y_train)

print("DONE!")

# Plot training MSE
plt.plot(history)
plt.show()

# Classify on Test set
print(">>>>>>>>>>Classifying on Test")
predictions = []

prediction = model.predict(X_test)
prediction_prob = model.predict(X_test, output_prob=True)

print("DONE!")

# Evaluate our model based on the performance on Test set
print(">>>>>>>>>>Evaluating")

accuracy = np.mean(y_test == prediction)
print(f'Accuracy is {accuracy*100:.4f}%')

#P rint out Confsuion matrix and Classification Report
print_results(prediction_prob,y_test)
print('The first 5 misclassified rows are as follows: ')
print('The format of these rows is: (X value: Y Predicted, Y Actual)')
print(get_misclassified_rows(X_test, prediction_prob, y_test)[:5])

print("DONE!")