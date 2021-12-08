import dataset as ds
import gensim_word2vec as w2v
import numpy as np
import itertools
from collections import OrderedDict
from multiprocessing import Pool

USE_MULTICORE = True
NUM_PROCS = 12

def worker_initializer():
    #Globals (need to be global for pooling to work)
    global WORDS_MAIN
    global WORD_LIBRARY
    global C
    global X

    osha_model = w2v.load_word2vec('osha_new_and_old')
    WORDS_MAIN = ds.load_words(ds.MAIN_DATA)
    WORD_LIBRARY = ds.get_library(WORDS_MAIN)
    C = w2v.get_C_mat(osha_model, WORD_LIBRARY)
    X = w2v.get_X_mat(osha_model, WORD_LIBRARY)

if __name__ == "__main__":
    worker_initializer()

def nBOW(doc):
    doc_unique, doc_counts = np.unique(doc, return_counts=True)
    word_dict = OrderedDict.fromkeys(WORD_LIBRARY, 0)
    for i, word in enumerate(doc_unique):
        word_dict[word] += doc_counts[i]
    total_words = np.sum(doc_counts)
    return np.array(list(word_dict.values())) / total_words

def WCD(nBOW1, nBOW2):
    return np.linalg.norm(X @ nBOW1 - X @ nBOW2)

def WMD(nBOW1, nBOW2):

    #TODO: SOLVE WMD HERE!
    return None

def relaxed_WMD(nBOW1, nBOW2):
    n = len(WORD_LIBRARY)
    cmin_j = np.argmin(C, axis=1)
    cmin_i = np.argmin(C, axis=0)

    dist = 0.0
    for i in range(n):
        for j in range(n):
            L1 = nBOW1[i] if j == cmin_j[i] else 0.0
            L2 = nBOW2[j] if i == cmin_i[j] else 0.0
            dist += np.maximum(L1, L2) * C[i, j]
    return dist

def WMD_matrix(doc_words, wmd_func=WMD, save_file="wmat.npy"):
    n = len(doc_words)
    W = np.zeros((n, n))

    nBOWs = [nBOW(d) for d in doc_words]
    for i in range(n):
        print("i:", i)
        if USE_MULTICORE:
            with Pool(NUM_PROCS, worker_initializer, ()) as p:
                row = p.starmap(wmd_func, zip(nBOWs, itertools.repeat(nBOWs[i])))
            W[i, :] = row
        else:
            for j in range(n):
                W[i, j] = wmd_func(nBOWs[i], nBOWs[j])

    with open(save_file, 'wb') as f:
        np.save(f, W)
    return W

def main():
    #RUN WCD MAT
    #print(WMD_matrix(WORDS_MAIN, wmd_func=WCD, save_file="WCD_wmat.npy"))

    #RUN RELAXED WMD
    print(WMD_matrix(WORDS_MAIN, wmd_func=relaxed_WMD, save_file="RWMD_wmat.npy"))

if __name__ == "__main__":
    main()