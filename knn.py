import numpy as np
import dataset as ds

def knn(k=5, train_split=0.8, file="WCD_wmat.npy"):
    num_train = int(train_split * 1000)

    with open(file, 'rb') as f:
        W = np.load(f)

    doc_ids = np.arange(num_train, W.shape[0])
    predictions = []

    labels, id_to_label = ds.load_labels()
    labels_train = np.array(labels[:num_train])
    labels_test = np.array(labels[num_train:])

    for w_i in doc_ids:
        training_dists = W[w_i, :num_train]

        # Finding the k nearest neighbors
        knn_idx = np.argpartition(training_dists, k)[:k]

        neighbors = {}
        for neighbor_label in labels_train[knn_idx]:
            if neighbor_label not in neighbors.keys():
                neighbors[neighbor_label] = 1
            else:
                neighbors[neighbor_label] += 1

        predictions.append(max(neighbors, key=neighbors.get))

    accuracy = 0
    for i, pred in enumerate(predictions):
        if pred == labels_test[i]:
            accuracy += 1
    accuracy /= W.shape[0] - num_train
    print(accuracy)

    return [id_to_label[pred] for pred in predictions]


def main():
    predictions = knn()

    print(predictions)

if __name__ == "__main__":
    main()