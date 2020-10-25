#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd

import os
import time
from collections import Counter
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


# In[17]:


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


# In[4]:


dataset_filename = os.listdir("./example_training/input")[0]
dataset_path = os.path.join(".","example_training/input",dataset_filename)
print("Open file:", dataset_path)
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)


# In[5]:


print("Dataset size:", len(df))


# In[6]:


df.head(5)


# In[7]:


decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]


# In[8]:


get_ipython().run_cell_magic('time', '', 'df.target = df.target.apply(lambda x: decode_sentiment(x))')


# In[14]:


target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")


# In[19]:


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


# In[20]:


get_ipython().run_cell_magic('time', '', 'df.text = df.text.apply(lambda x: preprocess(x))')


# In[21]:


df.head(5)


# In[ ]:




