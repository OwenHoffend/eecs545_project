import pandas as pd
import re

def get_words(df):
    #Filter out unwanted words (empty space, numeric characters, etc) here
    doc_words = []
    for doc in df['description']:
        words = re.findall(r"\S+(?=\s)", doc) #Split based on spaces, do not include spaces in the match
        #Insert other transforms here
        doc_words.append(words)
    return doc_words

def load_data():
    df = pd.read_csv('osha.csv')
    df.columns = ['id', 'summary', 'description', 'keywords', 'other']
    #Data preprocessing goes here
    return df


def main():
    #Test read OSHA dataset
    df = load_data()

if __name__ == "__main__":
    main()