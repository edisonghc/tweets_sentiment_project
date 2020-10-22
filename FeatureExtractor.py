from collections import Counter

class Features(object):

    def __init__(self):
        self.vocab = []
    
    def add_to_vocab(self, tweet):
        processed_tweet = self.preprocess(tweet)
        for word in processed_tweet:
            if not word in self.vocab:
                self.vocab.append(word.lower())

    def preprocess(self, tweet): 

        signs = [",",".","!","?",";","...","\"","--","..","#","@"]
        useless = ["and", "or","a","with","i","you","of","among","can","to","so","what","this","that","about","the","an","many","some","even","on","in","at","'s","-lrb-","-rrb-","are","`"]
        for word in tweet:
            if word in useless:
                tweet.remove(word)
                if word in signs:
                    tweet.remove(word)
        
        return tweet
        
    
    def extractor(self, tweet):
        processed_tweet = self.preprocess(tweet)
        ids = []
        for word in processed_tweet:
            if word.lower() in self.vocab:
                ids.append(self.vocab.index(word.lower()))

        return Counter(ids)

        

        
