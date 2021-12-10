import numpy as np
import dataset as ds

def knn(W_mat, num_test, labels_train, k=5):
    predictions = []

    for i in range(num_test):
        training_dists = W_mat[:, i]

        # Finding the k nearest neighbors
        knn_idx = np.argpartition(training_dists, k)[:k]

        neighbors = {}
        for neighbor_label in labels_train[knn_idx]:
            if neighbor_label not in neighbors.keys():
                neighbors[neighbor_label] = 1
            else:
                neighbors[neighbor_label] += 1

        predictions.append(max(neighbors, key=neighbors.get))

    return predictions

def calc_accuracy(predictions, labels_test):
    num_test = len(labels_test)

    accuracy = 0
    for i, pred in enumerate(predictions):
        if pred == labels_test[i]:
            accuracy += 1
    accuracy /= num_test
    return accuracy

def calc_precision(predictions, labels_test, num_classes):
    tp = 0
    fp = 0
    for label in range(num_classes):
        for i, pred in enumerate(predictions):
            if pred == label:
                if labels_test[i] == label:
                    tp += 1
                else:
                    fp += 1

    precision = tp / (tp + fp)
    return precision

def calc_recall(predictions, labels_test, num_classes):
    tp = 0
    fn = 0
    for label in range(num_classes):
        for i, pred in enumerate(predictions):
            if labels_test[i] == label:
                if pred == label:
                    tp += 1
                else:
                    fn += 1

    recall = tp / (tp + fn)
    return recall

def calc_f1(precision, recall):
    return 2 * (precision * recall)  (precision + recall)

def main():
    file = "WCD_wmat.npy"

    with open(file, 'rb') as f:
        W = np.load(f)

    W = np.zeros((800, 199)) # DELETE

    num_train, num_test = W.shape

    labels, id_to_label = ds.load_labels()
    labels_train = np.array(labels[:num_train])
    labels_test = np.array(labels[num_train:])
    
    num_classes = len(id_to_label.keys())


    predictions = knn(W, num_test, labels_train)
    accuracy = calc_accuracy(predictions, labels_test)
    print(accuracy)
    precision = calc_precision(predictions, labels_test, num_classes)
    print(precision)
    recall = calc_recall(predictions, labels_test, num_classes)
    print(recall)
    f1_score = calc_f1(precision, recall)
    print(f1_score)

    prediction_strings = [id_to_label[pred] for pred in predictions]


if __name__ == "__main__":
    main()