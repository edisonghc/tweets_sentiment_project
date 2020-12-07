"""
Authors: Edison Gu, Xinyue Li, Ang Li
    
"""

import numpy as np
import operator
from gensim.models import word2vec

class CountVectorizer:
    """
    Authors: Edison Gu, Xinyue Li

    Used in `main_classical.py'

    Mimic the behavior of:
        sklearn.feature_extraction.text.CountVectorizer¶
    """

    def __init__(self):
        self.vocab = np.array([])
        self.word2id = {}
        self.model = None
        
    def fit(self, documents, max_vocab=5000):
        """
        Learn the vocabulary dictionary. 

        Input:
            documents: ndarray (n_documents, )
        Output:
            list of most frequent word, and their unique ids
        """

        vocab_full = {}
    
        for text in documents:
            for word in text.split():
                vocab_full[word] = vocab_full.get(word, 0) + 1
        
        vocab_sorted = sorted(vocab_full.items(), key=operator.itemgetter(1), reverse=True)
        
        ideal_vocab_size = min(len(vocab_sorted), max_vocab)

        vocab_truncated = vocab_sorted[:ideal_vocab_size]

        vocab = np.array([k for k,_ in vocab_truncated])
        self.vocab = np.append(vocab, ['UNK'])
        self.word2id = dict([(word, id) for id, word in enumerate(self.vocab)])
    


    def fit_transform(self, documents, max_vocab=5000):
        """
        Learn the vocabulary dictionary and return document-term matrix.

        Input:
            documents: ndarray (n_documents, )
        Return:
            X: ndarray (n_documents, n_features)
        """

        self.fit(documents, max_vocab=max_vocab)

        n_documents = len(documents)
        n_features = len(self.vocab)
        unk_id = n_features - 1

        X = np.zeros((n_documents, n_features))

        for doc_id, text in enumerate(documents):
            for word in text.split():

                word_id = self.word2id.get(word, unk_id)
                X[doc_id, word_id] += 1

        return X
        

    def transform(self, documents):
        """
        Transform documents to document-term matrix.

        Input:
            documents: ndarray (n_documents, )
        Return:
            X: ndarray (n_documents, n_features)
        """
        
        n_documents = len(documents)
        n_features = len(self.vocab)
        unk_id = n_features - 1

        X = np.zeros((n_documents, n_features))

        for doc_id, text in enumerate(documents):
            for word in text.split():

                word_id = self.word2id.get(word, unk_id)
                X[doc_id, word_id] += 1

        return X
    
    def word2vec_init(self, tweet):
        """
        Authors: Ang Li

        Used in `main_classical.py'
    
        """

        W2V_SIZE = 128
        W2V_WINDOW = 5
        W2V_MIN_COUNT = 3
        W2V_EPOCH = 16
        documents= [s.split() for s in tweet]
        w2v_model = word2vec.Word2Vec(size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)
        w2v_model.build_vocab(documents)
        w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
        self.model = w2v_model
        
        embeded_tweets = []
        for t in documents:
            vec = np.zeros((W2V_SIZE))
            for word in t:
                if word in w2v_model.wv: 
                    vec += w2v_model.wv.vectors[w2v_model.wv.vocab[word].index]
            if(len(t)!=0): vec = vec/len(t)
            embeded_tweets.append(vec)
        return embeded_tweets
    
    def word2vec(self, tweet):
        W2V_SIZE = 128
        W2V_WINDOW = 5
        W2V_MIN_COUNT = 3
        W2V_EPOCH = 16
        
        documents= [s.split() for s in tweet]
        embeded_tweets = []
        for t in documents:
            vec = np.zeros((W2V_SIZE))
            for word in t:
                if word in self.model.wv: 
                    vec += self.model.wv.vectors[self.model.wv.vocab[word].index]
            if(len(t)!=0): vec = vec/len(t)
            embeded_tweets.append(vec)
        return embeded_tweets


class Features(object):
    """
    Author: Xinyue Li
    Bag of Words Feature Extractor
    """
    def __init__(self):
        self.train_vocab = []
    
    
    def add_to_vocab(self, processed_tweet):
        """
        Add valid words to the vocabulary
        :param processed_tweet: the processed tweet text
        """
        #iterate over the text
        for word in processed_tweet:
            #add only new words
            if not word in self.train_vocab:
                #do not add words containing digit
                if not any(char.isdigit() for char in word):
                    #do not add strange words of length greater than 15
                    if not len(word)>15:
                        self.train_vocab.append(word.lower())

        
    
    def extractor(self, processed_tweet, vocab):
        """
        feature extractor for bag-of-words embedding
        :param processed_tweet: the processed tweet text
        :param vocab: the vocabulary for feature extraction
        :return text_features: the feature vector of the given processed tweet 
        """
        #initialize the feature vector
        test_features = np.zeros(len(vocab))
        #iterate over the tweet text
        for word in processed_tweet:
            #find the index based on the word
            try:
                index = vocab.index(word)
            except ValueError:
                index = -1
            #update the count
            if index >= 0:
                test_features[index] = test_features[index]+1

        return test_features
