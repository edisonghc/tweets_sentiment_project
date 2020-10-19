
import numpy as np
import time
from collections import Counter


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        #get the indexer
        indexer = self.indexer
        #set up the feature vector
        feature = []
        #define some signs and useless words to avoid in indexer
        signs = [",",".","!","?",";","...","\"","--",".."]
        useless = ["and", "or","a","with","i","you","of","among","can","to","so","what","this","that","about","the","an","many","some","even","on","in","at","'s","-lrb-","-rrb-","are","`"]
        

        # for train set, add new valid words into the indexer
        if add_to_indexer:
            for word in ex_words:
                if not indexer.contains(word.lower()):
                    if not word in signs:
                        if not word.lower() in useless:
                            if not word.isdigit():
                                index = indexer.add_and_get_index(word.lower())

        # for test set, update the feature vector
        else:
            feature = np.zeros(len(indexer))
            for each in range(len(ex_words)):
                index = indexer.index_of(ex_words[each].lower())
                if index >= 0:
                    feature[index] = feature[index]+1
        

        return feature



class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...Combine unigram features with bigram features
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def add_and_count(self, in_words: List[str], this_word: str) -> bool:
        #check whether this word has occurred more than once
        if Counter(in_words)[this_word] > 1:
            return True
        else:
            return False
    
  
    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        #get the indexer
        indexer = self.indexer
        #set up the feature vector
        feature = []
        #define some signs and useless words to avoid in indexer
        signs = [",",".","!","?",";","...","\"","--",".."]
        useless = ["and", "or","a","with","i","you","of","among","can","from","to","so","what","who","whose","how","this","that","about","the","an","many","some","even","on","in","at","'s","-lrb-","-rrb-","are","`","here","there","his","her","with","without","has","have","became","become","becomes","above","under","even","does","for"]
        
        
        #unigram feature
        if add_to_indexer:
            for word in ex_words:
                if not indexer.contains(word.lower()):
                    if not word in signs:
                        if not word.lower() in useless:
                            if not word.isdigit():
                                index = indexer.add_and_get_index(word.lower())

        #bigram feature
        in_words = []
        if add_to_indexer:
            i = 0
            while i < len(ex_words)-1 :
                part1 = ex_words[i]
                #check whether this word is useful
                while part1 in signs or part1 in useless or part1.isdigit():
                    i = i+1
                    if i < len(ex_words):
                        part1 = ex_words[i]
                    else:
                        part1 = ""
                i = i+1
                part2 = ex_words[i]
                #check whether next word is useful
                while part2 in signs or part2 in useless or part2.isdigit():
                    i = i+1
                    if i < len(ex_words):
                        part2 = ex_words[i]
                    else:
                        part2 = ""
                word = part1+" "+part2
                in_words.append(word)
                #update the indexer
                if not indexer.contains(word.lower()):
                    #add only words occurring more than once
                    if self.add_and_count(in_words,word):
                        index = indexer.add_and_get_index(word.lower())

        else:
            #unigram part
            feature = np.zeros(len(indexer))
            for each in range(len(ex_words)):
                index = indexer.index_of(ex_words[each].lower())
                #update the feature vector
                if index >= 0:
                    feature[index] = feature[index]+1

            #bigram part
            words = []
            i = 0
            while i < len(ex_words)-1 :
                part1 = ex_words[i]
                #check whether this word is useful
                while part1 in signs or part1 in useless:
                    i = i+1
                    if i < len(ex_words):
                        part1 = ex_words[i]
                    else:
                        part1 = ""
                i = i+1
                part2 = ex_words[i]
                #check whether next word is useful
                while part2 in signs or part2 in useless:
                    i = i+1
                    if i < len(ex_words):
                        part2 = ex_words[i]
                    else:
                        part2 = ""
                #combine two words and update 
                word = part1+" "+part2
                words.append(word.lower())

            #update feature vector
            for each in words:
                index = indexer.index_of(each)
                if index >= 0:
                    feature[index] = feature[index]+1
        
        
        return feature



class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]





