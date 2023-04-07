import re
import json
import os
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Set the english stop words.
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

VECTORIZER_PATH = 'vectorizer_data'
def get_datapath(source_path):
    cwd = os.getcwd()
    parent_path = Path(cwd).parent
    # Set path for the raw data
    return parent_path / source_path


def json_loader(file_path):
    '''
        Takes a file path and loads a json line by line.
    '''
    
    with open(file_path) as f:
        lines = f.read().splitlines()
        
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)

    return pd.json_normalize(df_inter['json_element'].apply(json.loads))

def load_transformed_data(vectorizer_name):
    with open(
        get_datapath(VECTORIZER_PATH) / vectorizer_name / 'data.pkl', 'rb'
        ) as f:
        data = joblib.load(f)

        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_vectorizer(vectorizer_name):
    with open(
        get_datapath(VECTORIZER_PATH) / vectorizer_name / f'{vectorizer_name}.pkl', 'rb'
        ) as f:
        vectorizer = joblib.load(f)

    return vectorizer

def clean_lyrics(lyric):
    '''
        This function takes a str that represents each lyric
        and outputs a cleaned version of the lyric.
    '''
    # Lowercase lyrics and replace '\n' and apostrophes. 
    cleaned_lyric = lyric.lower().replace('\n', ' ').replace("\'",'')
    
    # Deal with tags.
    cleaned_lyric = re.sub(
        # (\[\w+\]) Matches any occurrence of [any word characters] or 
            # This matches all tags that are not the [Verse #] tag
        # (\[\w+ \d+\]) Matches any occurrence of [any word characters and any digit]
            # This is to match the [Verse #] tag
        '(\[(.*?)\])',
        '',
        cleaned_lyric
    )

    # Remove all punctuation.
    cleaned_lyric = re.sub(
        "[;:!><\"\',/@#$%&?–*+()|]",
        '',
        cleaned_lyric
    )
    
    # string.punctuation
    # Captures other punctuation that can not be replaced with the above regex.
    other_punctuation = ['-', '—','...', '”', '“', '.', '…', '’', '^', 'ι', '‘']
    for punctuation in other_punctuation:
        if punctuation == '—': 
            # This is to add spaces to large hyphens which denote breaks in the lyrics
            # not hyphenated words, so we want to separate these words.
            cleaned_lyric = cleaned_lyric.replace(punctuation, ' ')
        else:
            cleaned_lyric = cleaned_lyric.replace(punctuation, '')
  
    
    # Remove whitespaces greater than one. 
    cleaned_lyric = re.sub(
        ' {2,}', ' ',
        cleaned_lyric
    )
    return cleaned_lyric

def stop_word_removal_and_stem(lyrics):
    # Split the lyrics into a list where each index holds a word.
    tokenized_lyrics = lyrics.split(' ')

    # Try a SnowballStemmer that is less aggressive than the Porter stemmer. 
    stemmer = SnowballStemmer('english')

    # Stem the words if they are not a part of the ENGLISH_STOP_WORDS.
    lyrics_modified = [
        stemmer.stem(word) 
        for word in tokenized_lyrics
        if word not in ENGLISH_STOP_WORDS
    ]

    # Return a string with the modfied lyrics.
    return ' '.join(lyrics_modified)


def _convert_ada_embeddings(embedding):
    '''
    This function converts the ada embeddings in the dataset from a string
    to a numpy array.
    '''
    ada_embedding = embedding.replace('[', '').replace(']', '').replace(',', '')
    converted_ada_embedding = ada_embedding.split(' ')
    converted_ada_embedding = [
        float(weight)
        for weight in converted_ada_embedding 
    ]
    
    return np.array(converted_ada_embedding)

def get_ada_embeddings(embeddings):
    # Convert the embeddings from strings into an array. 
   
    ada_embeddings = [
        _convert_ada_embeddings(embedding)
        for embedding in embeddings
    ]
    # Stack the series and return the stacked array for modeling. 
    return np.stack(ada_embeddings, axis=0)
