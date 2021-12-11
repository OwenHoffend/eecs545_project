import dataset as ds
import gensim_word2vec as w2v
import numpy as np
from collections import OrderedDict

#User parameters (change these)
USE_MULTICORE = False
NUM_PROCS = 10
USE_GPU = True

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
    global OSHA_MODEL
    global WORDS_MAIN
    global WORD_LIBRARY
    global C
    global X
    global WL_SZ
    global LARGER_THAN_ALL_C

    OSHA_MODEL = w2v.load_word2vec('osha_wiki') #<-- New and old contains data from osha.csv and osha_new.csv (that's it)
    WORDS_MAIN = ds.load_words(ds.MAIN_DATA)
    WORD_LIBRARY = ds.get_library(WORDS_MAIN)
    C = w2v.get_C_mat(OSHA_MODEL, WORD_LIBRARY, save_file="cmat_wiki.npy")
    X = w2v.get_X_mat(OSHA_MODEL, WORD_LIBRARY, save_file="xmat_wiki.npy")
    WL_SZ = len(WORD_LIBRARY)
    if USE_GPU:
        C = torch.from_numpy(C).float().to(device)
        X = torch.from_numpy(X).float().to(device)
        LARGER_THAN_ALL_C = torch.max(C) + 1
    else:
        LARGER_THAN_ALL_C = np.max(C) + 1

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
    if USE_GPU:
        return torch.norm(X @ nBOW1 - X @ nBOW2).cpu().detach().numpy()
    else:
        return np.linalg.norm(X @ nBOW1 - X @ nBOW2)

def sparse_dense_mul(sparse, dense):
    i = sparse._indices()
    v = sparse._values()
    dv = dense[i[0,:], i[1,:]]
    return torch.sparse_coo_tensor(i, v * dv, sparse.size(), dtype=torch.float32, device=device)

def relaxed_WMD(nBOW1, nBOW2):
    if USE_GPU:
        d1_inds = torch.where(nBOW1 > 0)[0]
        d2_inds = torch.where(nBOW2 > 0)[0]

        #The following code enables us to take argmin along the rows/columns of C, while ignoring words from irrelevant documents
        #Consider that doing something like np.argmin(C[i, d1_inds]) would not work because the slicing reduces the size
        #of the resulting matrix, therefore we somehow need to exclude entires in C without removing them. I do this by
        #simply adding an amount that is guaranteed to be greater than everthing in C
        mask1 = torch.ones_like(C, dtype=torch.bool, device=device)
        mask1[d1_inds, :] = False
        mask2 = torch.ones_like(C, dtype=torch.bool, device=device)
        mask2[:, d2_inds] = False
        mask2 = torch.bitwise_or(mask1, mask2) * LARGER_THAN_ALL_C
        C_inf = torch.clone(C) + mask2
        
        cmin_j = torch.argmin(C_inf, dim=1)
        cmin_i = torch.argmin(C_inf, dim=0)

        t1_inds = torch.vstack((d1_inds, cmin_j[d1_inds]))
        t2_inds = torch.vstack((cmin_i[d2_inds], d2_inds))

        T1_cuda = torch.sparse_coo_tensor(t1_inds, nBOW1[d1_inds], (WL_SZ, WL_SZ), dtype=torch.float32, device=device)
        T2_cuda = torch.sparse_coo_tensor(t2_inds, nBOW2[d2_inds], (WL_SZ, WL_SZ), dtype=torch.float32, device=device)

        pWMD1 = torch.sparse.sum(sparse_dense_mul(T1_cuda, C))
        pWMD2 = torch.sparse.sum(sparse_dense_mul(T2_cuda, C))
        return torch.max(pWMD1, pWMD2).cpu().detach().numpy()
    else:
        d1_inds = np.where(nBOW1 > 0)[0]
        d2_inds = np.where(nBOW2 > 0)[0]

        mask1 = np.ones_like(C, dtype=np.bool_)
        mask1[d1_inds, :] = False
        mask2 = np.ones_like(C, dtype=np.bool_)
        mask2[:, d2_inds] = False
        mask2 = np.bitwise_or(mask1, mask2) * LARGER_THAN_ALL_C
        C_inf = np.copy(C) + mask2

        T1 = np.zeros((WL_SZ, WL_SZ), dtype=np.float32)
        T2 = np.zeros((WL_SZ, WL_SZ), dtype=np.float32)
        for i in d1_inds:
            j = np.argmin(C_inf[i, :])
            T1[i, j] = nBOW1[i]
        for j in d2_inds:
            i = np.argmin(C_inf[:, j])
            T2[i, j] = nBOW2[j]
        return np.maximum(np.sum(T1 * C), np.sum(T2 * C))

def WMD_matrix(train_docs, test_docs, wmd_func=WCD, save_file="wmat.npy"):
    ntrain = len(train_docs)
    ntest = len(test_docs)
    W = np.zeros((ntrain, ntest))

    nBOWs_train = np.array([nBOW(d) for d in train_docs], dtype=np.float32)
    nBOWs_test = np.array([nBOW(d) for d in test_docs], dtype=np.float32)
    if USE_GPU:
        nBOWs_train = torch.from_numpy(nBOWs_train).to(device)
        nBOWs_test = torch.from_numpy(nBOWs_test).to(device)
    for i in range(ntrain):
        print("i:", i)
        if USE_MULTICORE:
            with Pool(NUM_PROCS, worker_initializer, ()) as p:
                row = p.starmap(wmd_func, zip(itertools.repeat(nBOWs_train[i]), nBOWs_test))
            W[i, :] = row
        else:
            for j in range(ntest):
                #print("j:", j)
                W[i, j] = wmd_func(nBOWs_train[i], nBOWs_test[j])

    with open(save_file, 'wb') as f:
        np.save(f, W)
    return W

def gensim_WMD_matrix(train_docs, test_docs, save_file="WMD_wmat.npy"):
    ntrain = len(train_docs)
    ntest = len(test_docs)
    W = np.zeros((ntrain, ntest))
    for i in range(ntrain):
        print("i:", i)
        for j in range(ntest):
            W[i, j] = OSHA_MODEL.wv.wmdistance(train_docs[i], test_docs[j])
    with open(save_file, 'wb') as f:
        np.save(f, W)
    return W

def main():
    train_split = 0.8
    num_train = int(train_split * 1000)

    #RUN WCD MAT
    #print(WMD_matrix(WORDS_MAIN[:num_train], WORDS_MAIN[num_train:], wmd_func=WCD, save_file="WCD_wmat_wiki.npy"))

    #RUN RELAXED WMD
    #print(WMD_matrix(WORDS_MAIN[:num_train], WORDS_MAIN[num_train:], wmd_func=relaxed_WMD, save_file="RWMD_wmat_wiki.npy"))

    #RUN GENSIM WMD
    print(gensim_WMD_matrix(WORDS_MAIN[:num_train], WORDS_MAIN[num_train:], save_file="WMD_wmat_wiki.npy"))

if __name__ == "__main__":
    main()