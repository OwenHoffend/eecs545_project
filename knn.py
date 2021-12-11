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

def random_guess(num_test, num_classes):
    return np.random.randint(0, size=num_test, high=num_classes)

def random_guess_weighted(num_test, num_classes, weights):
    return np.random.choice(np.array(range(num_classes)), size=num_test, p=weights)

def confusion_matrix(predictions, labels_test, num_classes):
    Q = np.zeros((num_classes, num_classes))
    for i, pred in enumerate(predictions):
        Q[pred, labels_test[i]] += 1
    return Q

def calc_accuracy(predictions, labels_test):
    num_test = len(labels_test)

    accuracy = 0
    for i, pred in enumerate(predictions):
        if pred == labels_test[i]:
            accuracy += 1
    accuracy /= num_test
    return accuracy

def class_precisions(predictions, labels_test, num_classes):
    precisions = np.zeros(num_classes)
    for label in range(num_classes):
        tp = 0
        fp = 0
        for i, pred in enumerate(predictions):
            if pred == label: #Look at predictions that match this label
                if labels_test[i] == label:
                    tp += 1
                else:
                    fp += 1
        precisions[label] = tp / (tp + fp + 1e-6)
    return precisions

def class_recalls(predictions, labels_test, num_classes):
    recalls = np.zeros(num_classes)
    for label in range(num_classes):
        tp = 0
        fn = 0
        for i, pred in enumerate(predictions):
            if labels_test[i] == label: #look at examples that match this label
                if pred == label:
                    tp += 1
                else:
                    fn += 1
        recalls[label] = tp / (tp + fn + 1e-6)
    return recalls

def class_f1s(precisions, recalls):
    return 2 * (precisions * recalls) / (precisions + recalls + 1e-6) #1e-6 to prevent divide by 0

def weighted(metric, weights):
    return np.sum(metric * weights) / np.sum(weights)

def print_metrics(file, Q, metrics):
    print("File: {} Confusion Matrix: \n {}".format(file, Q))
    print("File: {} Accuracy: {}".format(file, metrics[0]))
    print("File: {} Weighted Precision: {}".format(file, metrics[1]))
    print("File: {} Weighted Recall: {}".format(file, metrics[2]))
    print("File: {} Weighted F1: {}".format(file, metrics[3]))

def get_all_metrics(file, labels, predictions, num_classes, num_train):
    labels_test = np.array(labels[num_train:])

    Q = confusion_matrix(predictions, labels_test, num_classes)
    accuracy = calc_accuracy(predictions, labels_test)
    precisions = class_precisions(predictions, labels_test, num_classes)
    recalls = class_recalls(predictions, labels_test, num_classes)
    f1_scores = class_f1s(precisions, recalls)

    _, weights = np.unique(labels, return_counts=True)
    metrics = np.array([ #Pack metrics for easy averaging (for random guess)
        weighted(accuracy, weights),
        weighted(precisions, weights),
        weighted(recalls, weights),
        weighted(f1_scores, weights)
    ])
    #Print confusion matrix and weighted metrics
    
    #prediction_strings = [id_to_label[pred] for pred in predictions]
    return Q, metrics

def main():
    file = "WCD_wmat_wiki.npy"
    with open(file, 'rb') as f:
        W = np.load(f)

    num_train, num_test = W.shape

    labels, id_to_label = ds.load_labels()
    labels_train = np.array(labels[:num_train])
    
    num_classes = len(id_to_label.keys())

    #kNN
    #predictions = knn(W, num_test, labels_train)
    #Q, metrics = get_all_metrics(file, labels, predictions, num_classes, num_train)
    #print_metrics(file, Q, metrics)

    #Random guess sampling
    NUM_SAMPLES = 100
    avg_metrics = np.zeros(4)
    __, weights = np.unique(labels, return_counts=True)
    for _ in range(NUM_SAMPLES):
        #Random guess - uniform
        predictions = random_guess(num_test, num_classes)

        #Random guess - weighted
        #predictions = random_guess_weighted(num_test, num_classes, weights / np.sum(weights))
        
        Q, metrics = get_all_metrics(file, labels, predictions, num_classes, num_train)
        avg_metrics += metrics
    avg_metrics /= NUM_SAMPLES
    print_metrics(file, Q, avg_metrics)

if __name__ == "__main__":
    main()