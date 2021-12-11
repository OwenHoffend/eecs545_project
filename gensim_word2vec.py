from gensim.models import Word2Vec
import gensim.downloader as api
import os
import numpy as np
import dataset as ds

VECTOR_SIZE = 50
USE_WIKI_WORDS = True #<-- Train w2v on a larger corpus? (Experimental)

def train_word2vec(doc_words, model_name='word2vec'):
    if USE_WIKI_WORDS:
        extra_words = api.load("wiki-english-20171001")
        model = Word2Vec(extra_words, min_count=1, vector_size=VECTOR_SIZE, workers=3, window=3, sg=1)
        model.build_vocab(doc_words, update=True)
        model.train(doc_words, total_examples=len(doc_words), epochs=1)
    else:
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
        print("Generating C mat. Saving to: ", save_file)
        C = np.zeros((n, n)) #May need to enforce m == n
        for i, w1 in enumerate(word_library):
            print("i:", i)
            for j, w2 in enumerate(word_library):
                v1 = model.wv.get_vector(w1, norm=True)
                v2 = model.wv.get_vector(w2, norm=True)
                C[i, j] = np.linalg.norm(v1 - v2)
        with open(save_file, 'wb') as f:
            np.save(f, C)
    else:
        print("Loading old C mat: ", save_file)
        with open(save_file, 'rb') as f:
            C = np.load(f)
    return C

def get_X_mat(model, word_library, save_file='xmat.npy'):
    d = VECTOR_SIZE
    n = len(word_library)

    if not os.path.exists(save_file):
        print("Generating X mat. Saving to: ", save_file)
        X = np.zeros((n, d))
        for i in range(n):
            X[i] = model.wv.get_vector(word_library[i], norm=True)
        with open(save_file, 'wb') as f:
            np.save(f, X)
    else:
        print("Loading old X mat: ", save_file)
        with open(save_file, 'rb') as f:
            X = np.load(f)

    return X.T #X is d x n

def main():
    words_w2v = ds.load_words(ds.W2V_DATA)
    words_main = ds.load_words(ds.MAIN_DATA)
    if USE_WIKI_WORDS:
        osha_model = train_word2vec(words_w2v + words_main, 'osha_wiki')
    else:
        osha_model = train_word2vec(words_w2v + words_main, 'osha_new_and_old')
    #osha_model = load_word2vec('osha')

    #Test the word vector model
    print(osha_model.wv.most_similar('fire')) #Returns things like "flames", "explosion", "extinguisher", etc

if __name__ == "__main__":
    main()