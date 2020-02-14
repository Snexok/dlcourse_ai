import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = np.isin(np.argwhere(prediction == True),  np.argwhere(ground_truth == True)).sum()
    tn = np.isin(np.argwhere(prediction == False), np.argwhere(ground_truth == False)).sum()
    fp = np.isin(np.argwhere(prediction == True),  np.argwhere(ground_truth == False)).sum()
    fn = np.isin(np.argwhere(prediction == False), np.argwhere(ground_truth == True)).sum()

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracies = []
    num_classes = np.unique(ground_truth)

    for i in num_classes:
        tp = np.isin(np.argwhere(prediction == i),  np.argwhere(ground_truth == i)).sum()
        tn = np.isin(np.argwhere(prediction != i), np.argwhere(ground_truth != i)).sum()
        fp = np.isin(np.argwhere(prediction == i),  np.argwhere(ground_truth != i)).sum()
        fn = np.isin(np.argwhere(prediction != i), np.argwhere(ground_truth == i)).sum()

        accuracy  = (tp + tn) / (tp + tn + fp + fn)

        accuracies.append(accuracy)
    return np.mean(accuracies)
