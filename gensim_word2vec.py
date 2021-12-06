from gensim.models import Word2Vec
import os
import pandas as pd
import numpy as np
import dataset

VECTOR_SIZE = 50

def train_word2vec(doc_words, model_name='word2vec'):
    model = Word2Vec(doc_words, min_count=1, vector_size=VECTOR_SIZE, workers=3, window=3, sg= 1)
    model.save(model_name + '.model')
    return model

def load_word2vec(model_name='word2vec'):
    return Word2Vec.load(model_name + ".model")

def get_C_mat(model, word_library, save_file='cmat.npy'):
    #Compute a matrix of the word2vec similarities of every pair of words taken from 2 documents
    #Get the c(i, j) matrix for a pair of documents, in order to perform WMD
    n = len(word_library)
    if not os.path.exists(save_file):
        C = np.zeros((n, n)) #May need to enforce m == n
        for i, w1 in enumerate(word_library):
            for j, w2 in enumerate(word_library):
                C[i, j] = model.wv.similarity(w1, w2)
        with open(save_file, 'wb') as f:
            np.save(f, C)
    else:
        with open(save_file, 'rb') as f:
            C = np.load(f)
    return C

def get_X_mat(model, word_library):
    d = VECTOR_SIZE
    n = len(word_library)
    X = np.zeros((n, d))
    for i in range(n):
        X[i] = model.wv[word_library[i]]
    return X.T #X is d x n

def main():
    df = dataset.load_data()
    doc_words = dataset.get_words(df)
    doc_library = dataset.get_library(df)
    #osha_model = train_word2vec(doc_words, 'osha')
    osha_model = load_word2vec('osha')

    #Test the word vector model
    print(osha_model.wv.most_similar('fire')) #Returns things like "flames", "explosion", "extinguisher", etc

    #Test word similarity matrix, C
    test_lib = ["is", "it", "the"]
    C = get_C_mat(osha_model, test_lib)
    print(C)

if __name__ == "__main__":
    main()