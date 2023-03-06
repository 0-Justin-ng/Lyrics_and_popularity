import re
import pandas as pd
import json
import os
from pathlib import Path
import joblib

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
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    return X_train, X_test, y_train, y_test

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

