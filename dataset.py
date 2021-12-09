import pandas as pd
import numpy as np
import regex
import itertools

W2V_DATA = {'csvfile' : 'osha.csv', 'columns' : ['id', 'summary', 'description', 'keywords', 'other']}
MAIN_DATA = {'csvfile' : 'osha_new.csv', 'columns' : ['summary', 'description', 'label']}

def get_library(doc_words):
    return np.unique(np.array(list(itertools.chain(*doc_words))))

def load_words(dfile_dict):
    df = pd.read_csv(dfile_dict['csvfile'])
    df.columns = dfile_dict['columns']
    #Filter out unwanted words (empty space, numeric characters, etc) here
    doc_words = []
    for doc in df['description']:
        # words = re.findall(r"\S+(?=\s)", doc) #Split based on spaces, do not include spaces in the match
        
        #Remove numbers
        #Remove punctuation
        #Remove common small words (maybe)
        # words = re.findall(r"[A-Za-z]{2,}", doc) # Find all >=2 letter words, no punctuation or numbers
        words = regex.findall(r"(?<=^| )(?!(?<!^|\. *)[A-Z][A-Za-z]+)[A-Za-z]{2,}", doc) # No mid-sentence proper nouns 

        #Make all lower case
        words = [word.lower() for word in words]
        doc_words.append(words)
    return doc_words

def load_labels(dfile_dict=MAIN_DATA):
    df = pd.read_csv(dfile_dict['csvfile'])
    df.columns = dfile_dict['columns']

    labels = []
    labels_dict = {}

    i = 0
    for label in df['label']:
        if label not in labels_dict.keys():
            labels_dict[label] = i
            i += 1

        labels.append(labels_dict[label])

    # Create label_id to label_name dict
    id_to_label = {v: k for k, v in labels_dict.items()}
    return (labels, id_to_label)

def main():
    #Test read OSHA dataset
    df = load_words(W2V_DATA)

if __name__ == "__main__":
    main()