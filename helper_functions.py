"""
Authors: Xinyue Li

    Additional helper functions for the program
    Append any function here to `main_classical.py` before using

"""

def write_output(weights, output_path):
    """
    Author: Xinyue Li
    write the weights data into the file, one element each line
    :param weights: an array of weights
    :param output_path: the path of the output file
    """
    #open the output file
    write_file = open(output_path, 'w')

    #write the data into the file
    for l in weights:
        write_file.write(str(l) + "\n")

    #close the file
    write_file.close()


def run_on_BoW_LR():
    """
    Author: Xinyue Li
    Train, test and evalute with the Bag-of-Words option and Logistic Regression model
    """
    print(">>>>>>>>>>Preprocessing")
    # apply preprocess method
    df.text = df.text.apply(lambda x: preprocess(x))
    print("DONE!")

    #tokenize
    print(">>>>>>>>>>Tokenizing")
    nltk.download('punkt')
    df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
    print("DONE!")

    #shuffle the data set and choose the training part and testing part
    df = df.sample(frac=1).reset_index(drop=True)
    tweets_for_training = df.head(1000000)
    tweets_for_testing = df.tail(1000)

    #build up the vocabulary
    print(">>>>>>>>>>Building Vocabulary")
    feature_collection = Features()
    for id in tweets_for_training.index:
        feature_collection.add_to_vocab(df['text'][id])
    print("DONE!")
    print("Vocabulary size: ", len(feature_collection.train_vocab))

    #train the model
    print(">>>>>>>>>>Training model")
    model = Logistic_Regression_for_BoW(tweets_for_training, feature_collection, feature_collection.train_vocab)
    print("DONE!")

    #write the weights and features to the output files
    write_output(model.weights, "output5.txt")
    write_output(feature_collection.train_vocab, "features5.txt")

    #classify
    print(">>>>>>>>>>Classifying")
    predictions = []
    for id in tweets_for_testing.index:
        predict = model.predict(df['text'][id], feature_collection)
        predictions.append(predict)
    print("DONE!")

    #evaluate the accuracy
    print(">>>>>>>>>>Evaluating")
    targets = tweets_for_testing['target']
    modified_targets = targets.replace(4, 1) # 4 is positive in the data file
    evaluation(modified_targets.values.tolist(), predictions)
    print("DONE!")

#run_on_BoW_LR

def test_on_BoW_LR():
    """
    Author: Xinyue Li
    Make the prediction on the testing data set basd on the trained LR model
    """
    # open and read data file
    print("Open file:", "ExtractedTweets.csv")
    df = pd.read_csv("ExtractedTweets.csv", encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

    #neglect the first row and shuffle the data
    df = df.iloc[1:]
    df = df.sample(frac=1).reset_index(drop=True)
    df_democrat = df.loc[df['party'] == "Democrat"]
    df_republican = df.loc[df['party'] == "Republican"]

    #print the size of the data set
    print("Dataset size:", len(df))
    print("Democrat dataset size", len(df_democrat))
    print("Republican dataset size", len(df_republican))

    print(">>>>>>>>>>Preprocessing")
    #apply preprocess method
    df.tweet = df.tweet.apply(lambda x: preprocess(x))
    df_democrat.tweet = df_democrat.tweet.apply(lambda x: preprocess(x))
    df_republican.tweet = df_republican.tweet.apply(lambda x: preprocess(x))
    print("DONE!")

    #tokenize
    print(">>>>>>>>>>Tokenizing")
    nltk.download('punkt')
    df['tweet'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
    print("DONE!")

    #read weights and features from the file
    weights = np.genfromtxt("output5.txt", delimiter=' ')
    features = []
    with open("features5.txt") as file_in:
        for feature in file_in:
            features.append(feature)

    #set up the LR model
    model = Logistic_Regression()
    model.weights = weights
    feature_extract = Features()
    feature_extract.train_vocab = features

    #classify and count results on democrats
    print(">>>>>>>>>>Classifying")
    predictions_d = []
    for id in df_democrat.index:
        predict = model.predict(df_democrat['tweet'][id], feature_extract)
        predictions_d.append(predict)
    pos_democrat = predictions_d.count(1)
    neg_democrat = predictions_d.count(0)

    #classify and count results on republicans
    predictions_r = []
    for id in df_republican.index:
        predict = model.predict(df_republican['tweet'][id], feature_extract)
        predictions_r.append(predict)
    pos_republican = predictions_r.count(1)
    neg_republican = predictions_r.count(0)

    #print out the prediction
    print("DONE!")
    print("Democrat : ",predictions_d)
    print("Democrat positive: ",pos_democrat)
    print("Democrat negative: ",neg_democrat)

    print("Republican: ", predictions_r)
    print("Republican positive: ",pos_republican)
    print("Republican negative: ",neg_republican)

#test_on_BoW_LR
