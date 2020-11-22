# from collections import Counter
import numpy as np
import operator

class CountVectorizer:
    """
    Mimic the behavior of:
        sklearn.feature_extraction.text.CountVectorizer¶
    """

    def __init__(self):
        self.vocab = np.array([])
        self.word2id = {}

    def fit(self, documents, max_vocab=5000):
        """
        Learn the vocabulary dictionary.

        Input:
            documents: ndarray (n_documents, )
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

# class Features(object):

#     def __init__(self):
#         self.train_vocab = []
    
    
#     def add_to_vocab(self, processed_tweet):
#         for word in processed_tweet:
#             if not word in self.train_vocab:
#                 self.train_vocab.append(word.lower())

        
    
#     def extractor(self, processed_tweet, vocab):
#         test_features = np.zeros(len(vocab))
#         ids = []
#         for word in processed_tweet:
#             try:
#                 index = vocab.index(word)
#             except ValueError:
#                 index = -1
#             if index >= 0:
#                 test_features[index] = test_features[index]+1

#         return test_features
