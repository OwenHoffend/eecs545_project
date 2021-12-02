from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import dataset

def train_word2vec(doc_words, model_name='word2vec'):
    model = Word2Vec(doc_words, min_count=1, vector_size=50, workers=3, window=3, sg= 1)
    model.save(model_name + '.model')
    return model

def load_word2vec(model_name='word2vec'):
    return Word2Vec.load(model_name + ".model")

def get_C_mat(doc1_words, doc2_words, model):
    #Compute a matrix of the word2vec similarities of every pair of words taken from 2 documents
    #Get the c(i, j) matrix for a pair of documents, in order to perform WMD
    m, n = len(doc1_words), len(doc2_words)
    C = np.zeros((m, n)) #May need to enforce m == n
    for i, w1 in enumerate(doc1_words):
        for j, w2 in enumerate(doc2_words):
            C[i, j] = model.wv.similarity(w1, w2)
    return C

def main():
    df = dataset.load_data()
    doc_words = dataset.get_words(df)
    osha_model = train_word2vec(doc_words, 'osha')
    #osha_model = load_word2vec('osha')

    #Test the word vector model
    print(osha_model.wv.most_similar('fire')) #Returns things like "flames", "explosion", "extinguisher", etc

    #Test word similarity matrix, C
    C = get_C_mat(doc_words[0], doc_words[1], osha_model)
    print(C)

if __name__ == "__main__":
    main()