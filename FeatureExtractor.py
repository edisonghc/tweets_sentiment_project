from collections import Counter
import numpy as np

class Features(object):

    def __init__(self):
        self.train_vocab = []
    
    
    def add_to_vocab(self, processed_tweet):
        for word in processed_tweet:
            if not word in self.train_vocab:
                self.train_vocab.append(word.lower())

        
    
    def extractor(self, processed_tweet, vocab):
        test_features = np.zeros(len(vocab))
        ids = []
        for word in processed_tweet:
            try:
                index = vocab.index(word)
            except ValueError:
                index = -1
            if index >= 0:
                test_features[index] = test_features[index]+1

        return test_features

        

        
