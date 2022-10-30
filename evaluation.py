import numpy as np

def confusion_matrix(predict_label, test_label, label_classes):
    confusion = np.zeros((len(label_classes), len(label_classes)))
    for i, label in enumerate(label_classes):
        # get predictions of current test label
        indices = (test_label == label)
        predictions = predict_label[indices]
        # get the counts per unique label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)
        frequency_dict = dict(zip(unique_labels, counts))
        # fill up the confusion matrix for the current row
        for j, class_label in enumerate(label_classes):
            confusion[i, j] = frequency_dict.get(class_label, 0)
    return confusion


def get_accuracy(confusion):
    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.


def get_precision(confusion):
    precisions = np.zeros((len(confusion),))
    for i in range(len(confusion)):
        # sum of count of this predicted label
        total_predict = np.sum(confusion[:, i])
        if total_predict > 0:
            precisions[i] = confusion[i, i] / total_predict
    return precisions


def get_recall(confusion):
    recalls = np.zeros((len(confusion),))
    for i in range(len(confusion)):
        # sum of count of this test label
        total_test = np.sum(confusion[i, :])
        if total_test > 0:
            recalls[i] = confusion[i, i] / total_test
    return recalls


def get_f1_measure(precision, recall):
    assert len(precision) == len(recall)
    f1_measures = np.zeros((len(precision),))
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_measures[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return f1_measures

def print_all_from_confusion_matrix(confusion):
    accuracy = get_accuracy(confusion)
    precision = get_precision(confusion)
    recall = get_recall(confusion)
    f1_measure = get_f1_measure(precision, recall)
    print(confusion)
    print("\tprecision               ", precision)
    print("\trecall                  ", recall)
    print("\tf1_measure              ", f1_measure)
    


# def evaluate(predict_label, test_label):
#     label_classes = np.unique(np.concatenate((predict_label, test_label)))
#     confusion = confusion_matrix(predict_label, test_label, label_classes)
#     accuracy = get_accuracy(confusion)
#     precision = get_precision(confusion)
#     recall = get_recall(confusion)
#     f1_measure = get_f1_measure(precision, recall)
#     print("confusion matrix: \n", confusion)
#     print("accuracy: ", accuracy)
#     print("precision: ", precision)
#     print("recall: ", recall)
#     print("f1_measure: ", f1_measure)
