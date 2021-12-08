import dataset
import gensim_word2vec as w2v
import numpy as np
from collections import OrderedDict

def nBOW(doc, word_library):
    doc_unique, doc_counts = np.unique(doc, return_counts=True)
    word_dict = OrderedDict.fromkeys(word_library, 0)
    for i, word in enumerate(doc_unique):
        word_dict[word] += doc_counts[i]
    total_words = np.sum(doc_counts)
    return np.array(list(word_dict.values())) / total_words

def WCD(doc1, doc2, word_library, w2v_model, X):
    nBOW1 = nBOW(doc1, word_library)
    nBOW2 = nBOW(doc2, word_library)
    
    return np.linalg.norm(X @ nBOW1 - X @ nBOW2)

def WMD(doc1, doc2, word_library, w2v_model, C):
    nBOW1 = nBOW(doc1, word_library)
    nBOW2 = nBOW(doc2, word_library)
    # C = w2v.get_C_mat(w2v_model, word_library)

    #TODO: SOLVE WMD HERE!
    return None

def relaxed_WMD(doc1, doc2, word_library, w2v_model, C):
    nBOW1 = nBOW(doc1, word_library)
    nBOW2 = nBOW(doc2, word_library)
    # C = w2v.get_C_mat(w2v_model, word_library)
    n = len(word_library)

    cmin_j = np.argmin(C, axis=1)
    cmin_i = np.argmin(C, axis=0)

    dist = 0.0
    for i in range(n):
        for j in range(n):
            L1 = nBOW1[i] if j == cmin_j[i] else 0.0
            L2 = nBOW2[j] if i == cmin_i[j] else 0.0
            dist += np.maximum(L1, L2) * C[i, j]
    return dist

def WMD_matrix(doc_words, word_library, w2v_model, wmd_func=WMD, input_mat=None, save_file="wmat.npy"):
    n = len(doc_words)
    W = np.zeros((n, n))
    for i, d1 in enumerate(doc_words):
        print("i:", i)
        for j, d2 in enumerate(doc_words):
            print("j:", j)
            W[i, j] = wmd_func(d1, d2, word_library, w2v_model, input_mat)

    with open(save_file, 'wb') as f:
        np.save(f, W)
    return W #<-- Perform clustering using this

def main():
    df = dataset.load_data()
    doc_words = dataset.get_words(df)

    #Get the library of words that are present in all documents
    word_library = dataset.get_library(df)
    osha_model = w2v.load_word2vec('osha') #<-- can also do train_word2vec here

    #Test WMD
    #WMD(doc_words[0], doc_words[1], word_library, osha_model)

    #print(WCD(doc_words[0], doc_words[1], word_library, osha_model))
    # print(relaxed_WMD(doc_words[0], doc_words[1], word_library, osha_model))
    X = w2v.get_X_mat(osha_model, word_library)
    print(WMD_matrix(doc_words, word_library, osha_model, wmd_func=WCD, input_mat=X))

if __name__ == "__main__":
    main()