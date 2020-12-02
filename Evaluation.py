from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np

def evaluation(targets, predictions):
    """
    Author: Xinyue Li
    Prints evaluation statistics comparing targets and predictions
    """
    num_correct = 0
    num_total = 0
    if len(targets) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(targets), len(predictions)))
    #iterate over all results
    for id in range(0, len(targets)):
        #check the num of correct prediction
        target = targets[id]
        prediction = predictions[id]
        if prediction == target:
            num_correct += 1
        num_total += 1
    #print the results
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

#Author: Simon Manning
def print_results(y_score,y_true):
    assert (len(y_score) == len(y_true)),'The number of inputs does not match the number of outputs'
    ### How to read the confusion_matrix
    print('Rows represent the predicted class and columns represent the actual class')
    y_pred = np.array(y_score > 0.5, dtype=int)
    print(f"Confusion Matrix: \n {confusion_matrix(y_true,y_pred)}")
    print('Classification Report')
    ### You can pass another variable: target_names to get a labeled result - this is recommended
    print(f" {classification_report(y_true,y_pred)}")

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return None

#Author: Simon Manning
def get_misclassified_rows(x, y_pred, y_test):
    triples = zip(list(x,y_pred,y_test))
    return [i for i in triples if (i[1] != i[2])]
