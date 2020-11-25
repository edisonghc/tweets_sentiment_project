import numpy as np
# import pandas as pd
# from math import e

class LogisticRegression:
    """
    Mimic the behavior of:
        sklearn.linear_model.LogisticRegression
    """

    def __init__(self):
        self.weights = np.array([])

    def init_weight(self, num_weight):

        # self.weights = np.zeros(num_weight)
        self.weights = np.random.uniform(low=-1, high=1, size=num_weight)


    def predict(self, X, added_bias=False, output_prob=False):
        """
        :param
            X: ndarray (n_samples, n_features)
            output_prob: return binary class label or probability of positive class, default True
        :return: [0,1] for neg or pos class OR probability of positive class
        """

        # if self.added_bias:
        #     input = np.append([1],X)

        if added_bias:
            input = X
        else:
            input = np.column_stack((np.ones(len(X)), X))

        # Get the sigmoid function
        ayda = input @ self.weights
        prob = 1 / (1 + np.exp(-1 * ayda))

        # Return probabilities or class labels
        if output_prob:
            return prob
        else:
            return np.array(prob > 0.5, dtype=int)


    ### Adding in confusion matrix to end of fit method
    def fit(self, X, target, learning_rate=0.3, max_epoch=100, tolerance=1E-3):
        """
        Train a logistic regression model.
        """

        epoch_size = len(X)
        history = np.array([])

        # Initialize weight
        self.init_weight(X.shape[1] + 1)

        input = np.column_stack((np.ones(len(X)), X))

        # Train the training set for specific times
        for _ in range(max_epoch):

            # Train the set in a random order
            order = np.random.choice(range(epoch_size), size=epoch_size, replace=False)
            epoch_error = 0

            for i in order:

                # Update the weight vector through the gradient according to the loss
                y_pred = self.predict(input[i], added_bias=True, output_prob=True)

                error = target[i] - y_pred
                epoch_error += error ** 2

                gradient = error * input[i]
                self.weights = self.weights + learning_rate * gradient

            # Log the loss
            epoch_error /= epoch_size
            history = np.append(history, epoch_error)

            # Early stopping when mean loss of an epoch is smaller than a tolerance
            if epoch_error < tolerance:
                break

        return history

    # def predict_2(self, tweet, feature_extractor):
    #     """
    #     :param ex_words: words (List[str]) in the sentence to classify
    #     :return: Either 0 for negative class or 1 for positive class
    #     """

    #     vocab = feature_extractor.train_vocab

    #     #get the feature vector by extractor
    #     feature_vector = feature_extractor.extractor(tweet, vocab)

    #     #get the sigmoid function
    #     power = np.sum(np.multiply(self.weights[1:], feature_vector)) + self.weights[0]
    #     prob = 1/(1+np.exp(-1*power))
    #     #evaluate the probability to give a result
    #     result = 0
    #     if prob > 0.5:
    #         result = 1

    #     return result

    # def train_LR(tweets_for_training, feature_extractor, vocab):
    #     """
    #     Train a logistic regression model.
    #     """
    #     # learning rate and training time for different feature extractor
    #     learning_rate = 0.8
    #     training_num = 10



    #     #set up logistic regression model
    #     model = Logistic_Regression()
    #     model.init_weight(len(vocab)+1)
    #     weight = model.weights


    #     # train the training set for specific times
    #     for t in range(training_num):
    #         #train the set in a random order
    #         shuffled = tweets_for_training.sample(frac = 1).reset_index(drop=True)
    #         for i in range(shuffled.shape[0]-1):
    #             #update the weight vector through the gradient according to the loss
    #             feature_vector = feature_extractor.extractor(shuffled.iloc[i].text, vocab)
    #             power = np.sum(np.multiply(feature_vector,weight[1:]))+weight[0]
    #             predict_prob = 1/(1+e**(-1*power))
                # difference = predict_prob-(shuffled.iloc[i].target/4)
                # gradient = np.multiply(feature_vector, difference)
                # gradient = np.append(difference,gradient)
    #             weight = weight - gradient * learning_rate
    #             model.weight_vector = weight

    #     return model
