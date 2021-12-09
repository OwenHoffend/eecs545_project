import dataset as ds
import gensim_word2vec as w2v
import numpy as np
from collections import OrderedDict

#User parameters (change these)
USE_MULTICORE = False
NUM_PROCS = 12
USE_GPU = False

#Global configuration (don't change these)
USE_MULTICORE = USE_MULTICORE and not USE_GPU #Don't try to use both
if USE_MULTICORE:
    from multiprocessing import Pool
    import itertools
if USE_GPU:
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def worker_initializer():
    #Globals (need to be global for pooling to work)
    global WORDS_MAIN
    global WORD_LIBRARY
    global C
    global CMIN_J
    global CMIN_I
    global X

    osha_model = w2v.load_word2vec('osha_new_and_old')
    WORDS_MAIN = ds.load_words(ds.MAIN_DATA)
    WORD_LIBRARY = ds.get_library(WORDS_MAIN)
    C = w2v.get_C_mat(osha_model, WORD_LIBRARY)
    if USE_GPU:
        CMIN_J = torch.from_numpy(np.argmin(C, axis=1)).to(device)
        CMIN_I = torch.from_numpy(np.argmin(C, axis=0)).to(device)
        C = torch.from_numpy(C).float().to(device)
    else:
        CMIN_J = np.argmin(C, axis=1)
        CMIN_I = np.argmin(C, axis=0)
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
    if USE_GPU:
        T1_cuda = torch.cuda.FloatTensor(n, n).fill_(0)
        T2_cuda = torch.cuda.FloatTensor(n, n).fill_(0)
        T1_cuda[:, CMIN_J] = nBOW1
        T2_cuda[CMIN_I, :] = nBOW2
        return torch.sum(torch.maximum(T1_cuda, T2_cuda) * C).cpu().detach().numpy()
    else:
        T1 = np.zeros((n, n))
        T2 = np.zeros((n, n))
        T1[:, CMIN_J] = nBOW1
        T2[CMIN_I, :] = nBOW2
        return np.sum(np.maximum(T1, T2) * C)

def WMD_matrix(doc_words, wmd_func=WMD, save_file="wmat.npy"):
    n = len(doc_words)
    W = np.zeros((n, n))

    nBOWs = np.array([nBOW(d) for d in doc_words])
    if USE_GPU:
        nBOWs = torch.from_numpy(nBOWs).float().to(device)
    for i in range(n):
        print("i:", i)
        if USE_MULTICORE:
            with Pool(NUM_PROCS, worker_initializer, ()) as p:
                row = p.starmap(wmd_func, zip(itertools.repeat(nBOWs[i]), nBOWs))
            W[i, :] = row
        else:
            for j in range(n):
                print("j:", j)
                W[i, j] = wmd_func(nBOWs[i], nBOWs[j])
        print('hi')

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