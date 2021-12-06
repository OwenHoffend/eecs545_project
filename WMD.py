import dataset
import gensim_word2vec as w2v
import itertools
import numpy as np
from collections import OrderedDict

def nBOW(doc, word_library):
    doc_unique, doc_counts = np.unique(doc, return_counts=True)
    word_dict = OrderedDict.fromkeys(word_library, 0)
    for i, word in enumerate(doc_unique):
        word_dict[word] += doc_counts[i]
    total_words = np.sum(doc_counts)
    return np.array(list(word_dict.values())) / total_words

def WCD(doc1, doc2, word_library, w2v_model):
    nBOW1 = nBOW(doc1, word_library)
    nBOW2 = nBOW(doc2, word_library)
    X = w2v.get_X_mat(w2v_model, word_library)

    return np.linalg.norm(X @ nBOW1 - X @ nBOW2)

def WMD(doc1, doc2, word_library, w2v_model):
    nBOW1 = nBOW(doc1, word_library)
    nBOW2 = nBOW(doc2, word_library)
    C = w2v.get_C_mat(doc1, doc2, w2v_model)

    #TODO: SOLVE WMD HERE!

    return None

def WMD_matrix(doc_words, word_library, w2v_model):
    n = len(doc_words)
    W = np.zeros((n, n))
    for i, d1 in enumerate(doc_words):
        for j, d2 in enumerate(doc_words):
            W[i, j] = WMD(d1, d2, word_library, w2v_model)
    return W #<-- Perform clustering using this

def main():
    df = dataset.load_data()
    doc_words = dataset.get_words(df)

    #Get the library of words that are present in all documents
    word_library = dataset.get_library(df)
    osha_model = w2v.load_word2vec('osha') #<-- can also do train_word2vec here

    #Test WMD
    WMD(doc_words[0], doc_words[1], word_library, osha_model)

    print(WCD(doc_words[0], doc_words[1], word_library, osha_model))

if __name__ == "__main__":
    main()