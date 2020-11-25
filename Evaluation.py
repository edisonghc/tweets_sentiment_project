

def evaluation(targets, predictions):
    """
    Prints evaluation statistics comparing targets and predictions,
    """
    num_correct = 0
    num_total = 0
    if len(targets) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(targets), len(predictions)))
    for id in range(0, len(targets)):
        target = targets[id]
        prediction = predictions[id]
        if prediction == target:
            num_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def print_results(y_pred,y_true):
    assert len(y_pred) == len(y_true),'The number of inputs does not match the number of outputs'
    ### How to read the confusion_matrix
    print('Rows represent the predicted class and columns represent the actual class')
    print(f"Confusion Matrix: {confusion_matrix(y_true,y_pred)}")
    print('Classification Report')
    ### You can pass another variable: target_names to get a labeled result - this is recommended
    print(f"Classification Report: {classification_report(y_true,y_pred)}")
    return None
