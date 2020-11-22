

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
    