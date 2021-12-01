import pandas as pd
import re

def load_data():
    df = pd.read_csv('osha.csv')
    df.columns = ['id', 'summary', 'description', 'keywords', 'other']
    #Data preprocessing goes here
    return df

def get_words(doc):
    #Filter out unwanted words (empty space, numeric characters, etc) here
    words = re.findall(r"\S+(?=\s)", doc) #Split based on spaces, do not include spaces in the match
    #Insert other transforms here
    return words

def main():
    #Test read OSHA dataset
    df = load_data()

if __name__ == "__main__":
    main()