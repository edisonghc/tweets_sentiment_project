from collections import Counter

class Features(object):

    def __init__(self):
        self.vocab = []
    
    def add_to_vocab(self, processed_tweet):
        for word in processed_tweet:
            if not word in self.vocab:
                self.vocab.append(word.lower())

        
    
    def extractor(self, processed_tweet):
        ids = []
        for word in processed_tweet:
            if word.lower() in self.vocab:
                ids.append(self.vocab.index(word.lower()))

        return Counter(ids)

        

        
