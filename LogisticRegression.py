import numpy as np
import pandas as pd
from math import e

class Logistic_Regression(object):
    def __init__(self):
        self.weights = np.array([])
        

    def init_weight(self, num_weight):
        self.weights = np.zeros(num_weight)
    
    def predict(self, tweet, feature_extractor):
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        vocab = feature_extractor.train_vocab
        #get the feature vector by extractor
        feature_vector = feature_extractor.extractor(tweet, vocab)
        #get the sigmoid function 
        power = np.sum(np.multiply(self.weights[1:], feature_vector)) + self.weights[0]
        prob = 1/(1+e**(-1*power))
        #evaluate the probability to give a result
        result = 0
        if prob > 0.5:
            result = 1

        return result

def train_LR(tweets_for_training, feature_extractor, vocab):
    """
    Train a logistic regression model.
    """
    # learning rate and training time for different feature extractor
    learning_rate = 0.8
    training_num = 10



    #set up logistic regression model
    model = Logistic_Regression()
    model.init_weight(len(vocab)+1)
    weight = model.weights


    # train the training set for specific times
    for t in range(training_num): 
        #train the set in a random order    
        shuffled = tweets_for_training.sample(frac = 1).reset_index(drop=True)
        for i in range(shuffled.shape[0]-1):
            #update the weight vector through the gradient according to the loss 
            feature_vector = feature_extractor.extractor(shuffled.iloc[i].text, vocab)
            power = np.sum(np.multiply(feature_vector,weight[1:]))+weight[0]
            predict_prob = 1/(1+e**(-1*power))
            difference = predict_prob-(shuffled.iloc[i].target/4)
            gradient = np.multiply(feature_vector, difference)
            gradient = np.append(difference,gradient)
            weight = weight - gradient * learning_rate
            model.weight_vector = weight
            
    return model
